"""Entrenamiento reproducible del Problema 2; el test se consulta solo al final.
Ejecutar: python entrenar.py. Nunca se ajustan umbrales con el test manual.
"""
import os
for k,v in {'TF_CPP_MIN_LOG_LEVEL':'3','TF_ENABLE_ONEDNN_OPTS':'0','CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}.items(): os.environ[k]=v
import hashlib, io, json, time, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, make_scorer, precision_recall_fscore_support, multilabel_confusion_matrix
from texto import TEMAS, SEMILLA, preparar, etiquetas_reglas, limpiar
from keras_red import RedKeras
P=Path(__file__).parent
MAX_EPOCAS=100; PACIENCIA=6; MIN_DELTA=.0001; BATCH=128
try:
 tf.config.threading.set_inter_op_parallelism_threads(1); tf.config.threading.set_intra_op_parallelism_threads(2)
except RuntimeError: pass
tf.config.experimental.enable_op_determinism()

def f1(y,p): return float(f1_score(y,p,average='macro',zero_division=0))
def bce(y,p):
 p=np.clip(p,1e-7,1-1e-7); return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def lotes(x,y,entrenamiento=False):
 ds=tf.data.Dataset.from_tensor_slices((x.astype('float32'),y.astype('float32')))
 if entrenamiento: ds=ds.shuffle(len(x),seed=SEMILLA,reshuffle_each_iteration=True)
 opt=tf.data.Options(); opt.threading.private_threadpool_size=1; opt.threading.max_intra_op_parallelism=1
 return ds.batch(BATCH).with_options(opt)
def crear_red(n,capas,dropout,pesos=None):
 tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEMILLA)
 red=tf.keras.Sequential([tf.keras.layers.Input(shape=(n,),name='texto_denso')],name='mlp_resenas')
 for i,u in enumerate(capas):
  red.add(tf.keras.layers.Dense(u,activation='relu',name=f'oculta_{i+1}'))
  if dropout: red.add(tf.keras.layers.Dropout(dropout,name=f'dropout_{i+1}'))
 red.add(tf.keras.layers.Dense(5,activation='sigmoid',name='cinco_temas'))
 loss='binary_crossentropy'
 if pesos is not None:
  w=tf.constant(pesos,dtype=tf.float32)
  def loss(y,p):
   p=tf.clip_by_value(p,1e-7,1-1e-7)
   return tf.reduce_mean(-(w*y*tf.math.log(p)+(1-y)*tf.math.log(1-p)),axis=-1)
 red.compile(optimizer=tf.keras.optimizers.Adam(.001),loss=loss); return red

def ajustar_red(x,y,xv,yv,config,controles=True):
 inicio=time.perf_counter(); pre=preparar(red=True,componentes=config['svd'],max_features=6000)
 z=pre.fit_transform(x).astype('float32'); zv=pre.transform(xv).astype('float32')
 # Pesos por etiqueta calculados exclusivamente con entrenamiento.
 pesos=np.sqrt((len(y)-y.sum(axis=0))/np.maximum(y.sum(axis=0),1)) if config['ponderada'] else None
 red=crear_red(z.shape[1],tuple(config['capas']),config['dropout'] if controles else 0,pesos)
 cb=tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=PACIENCIA,min_delta=MIN_DELTA,restore_best_weights=True)
 h=red.fit(lotes(z,y,True),validation_data=lotes(zv,yv),epochs=MAX_EPOCAS,callbacks=[cb] if controles else [],shuffle=False,verbose=0).history
 m=RedKeras(pre,red); pv=m.predict_proba(xv)
 info={**config,'F1 validación automática':f1(yv,pv>=.5),'epocas_ejecutadas':len(h['loss']),'mejor_epoca':int(cb.best_epoch+1) if controles else MAX_EPOCAS,'parada_anticipada':bool(cb.stopped_epoch>0) if controles else False,'early_stopping_activo':controles,'entrenamiento_s':time.perf_counter()-inicio,'val_loss_restaurada':bce(yv,pv),'pesos_positivos':pesos.tolist() if pesos is not None else [1]*5,'svd_varianza_explicada':float(pre['columnas'].named_transformers_['texto']['svd'].explained_variance_ratio_.sum())}
 return m,h,info

def exportar(m):
 p=P/'modelos'; p.mkdir(exist_ok=True); buf=io.BytesIO(); joblib.dump(m.preparacion,buf,compress=3); raw=buf.getvalue(); parts=[]
 for old in p.glob('keras_pre.part*'): old.unlink()
 for i,start in enumerate(range(0,len(raw),350000)):
  name=f'keras_pre.part{i:02d}'; block=raw[start:start+350000]; (p/name).write_bytes(block)
  parts.append({'archivo':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
 # La pérdida ponderada es necesaria solo para entrenar. Exportar configuración estándar
 # conserva exactamente los pesos; la inferencia usa compile=False.
 m.red.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy'); m.red.save(p/'red_keras.keras')
 (p/'keras_manifest.json').write_text(json.dumps({'tensorflow':tf.__version__,'preparacion_parts':parts,'preparacion_sha256':hashlib.sha256(raw).hexdigest(),'red_sha256':hashlib.sha256((p/'red_keras.keras').read_bytes()).hexdigest()},indent=2))

def particion(d,m):
 assert not set(d.listing_id)&set(m.listing_id),'Alojamientos del test en desarrollo'
 assert not set(d.comments.map(limpiar))&set(m.comments.map(limpiar)),'Texto del test en desarrollo'
 a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEMILLA).split(d,groups=d.listing_id))
 assert not set(d.iloc[a].listing_id)&set(d.iloc[b].listing_id)
 return a,b

def main():
 d=pd.read_csv(P/'datos/entrenamiento_debil.csv.gz',dtype={'listing_id':str},keep_default_na=False)
 m=pd.read_csv(P/'datos/muestra_manual.csv',dtype={'listing_id':str},keep_default_na=False)
 # Regenerar etiquetas tras corregir reglas lingüísticas; nunca modificar etiquetas humanas.
 d[TEMAS]=[etiquetas_reglas(t) for t in d.comments]
 d.to_csv(P/'datos/entrenamiento_debil.csv.gz',index=False,compression='gzip')
 a,b=particion(d,m); x=d.iloc[a][['comments']]; y=d.iloc[a][TEMAS].to_numpy(int); g=d.iloc[a].listing_id.to_numpy()
 xv=d.iloc[b][['comments']]; yv=d.iloc[b][TEMAS].to_numpy(int)
 cv=list(GroupKFold(3,shuffle=True,random_state=SEMILLA).split(x,y,g)); score=make_scorer(f1_score,average='macro',zero_division=0)
 modelos={}; costos={}; busquedas=[]; cvrows=[]
 for nombre,est in [('LogisticRegression',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=1000,random_state=SEMILLA)),('LinearSVC',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=SEMILLA))]:
  pipe=Pipeline([('preparacion',preparar()),('clasificador',OneVsRestClassifier(est))]); inicio=time.perf_counter()
  search=GridSearchCV(pipe,{'clasificador__estimator__C':[.5,2],'preparacion__texto__ngram_range':[(1,1),(1,2)]},scoring=score,cv=cv,n_jobs=1)
  search.fit(x,y); modelos[nombre]=search.best_estimator_; costos[nombre]=time.perf_counter()-inicio
  joblib.dump(modelos[nombre],P/(nombre+'.joblib'),compress=3)
  busquedas.append({'modelo':nombre,'F1 CV automático':search.best_score_,'parametros':search.best_params_,'F1 validación automática':f1(yv,modelos[nombre].predict(xv))})
  print('CLASICO',busquedas[-1],flush=True)
  ingenua=list(KFold(3,shuffle=True,random_state=SEMILLA).split(x))
  for validador,pliegues in [('KFold ingenuo',ingenua),('GroupKFold correcto',cv)]:
   valores=cross_val_score(clone(modelos[nombre]),x,y,cv=pliegues,scoring=score)
   for k,((ia,ib),v) in enumerate(zip(pliegues,valores),1):
    cvrows.append({'modelo':nombre,'validador':validador,'pliegue':k,'Macro F1 automático':v,'grupos_compartidos':len(set(g[ia])&set(g[ib]))})
 pd.DataFrame(cvrows).to_csv(P/'datos/particiones.csv',index=False)
 (P/'datos/busquedas.json').write_text(json.dumps(busquedas,indent=2))
 candidatos=[]; inicio=time.perf_counter()
 configs=[{'svd':64,'capas':[64],'dropout':.2,'ponderada':False},{'svd':128,'capas':[128,64],'dropout':.2,'ponderada':True},{'svd':256,'capas':[128,64],'dropout':.2,'ponderada':True}]
 for config in configs:
  red,h,info=ajustar_red(x,y,xv,yv,config); candidatos.append((red,h,info)); print('KERAS',info,flush=True)
 costos['MLP']=time.perf_counter()-inicio
 red,h,e=max(candidatos,key=lambda c:c[2]['F1 validación automática']); modelos['MLP']=red; exportar(red)
 pd.DataFrame([c[2] for c in candidatos]).to_csv(P/'datos/arquitecturas.csv',index=False)
 pd.DataFrame({'epoca':range(1,len(h['loss'])+1),'Entrenamiento':h['loss'],'Validación':h['val_loss']}).to_csv(P/'datos/perdida_keras.csv',index=False)
 sin,hb,eb=ajustar_red(x,y,xv,yv,e,False)
 pd.DataFrame({'epoca':range(1,len(hb['loss'])+1),'Entrenamiento':hb['loss'],'Validación':hb['val_loss']}).to_csv(P/'datos/perdida_sin_control.csv',index=False)
 evidencia={**e,'loss_train_restaurada':bce(y,red.predict_proba(x)),'loss_val_restaurada':bce(yv,red.predict_proba(xv)),'loss_train_sin_control':bce(y,sin.predict_proba(x)),'loss_val_sin_control':bce(yv,sin.predict_proba(xv)),'F1_val_sin_control':eb['F1 validación automática'],'max_epocas':MAX_EPOCAS,'patience':PACIENCIA,'min_delta':MIN_DELTA,'arquitectura_oculta':e['capas'],'entrada_dimensiones':e['svd']+1,'capas_densas':len(e['capas'])+1,'salida_unidades':5,'activacion_oculta':'relu','activacion_salida':'sigmoid','optimizador':'Adam','learning_rate':.001,'loss':'binary crossentropy ponderada' if e['ponderada'] else 'binary crossentropy','batch_size':BATCH,'tensorflow':tf.__version__,'keras':tf.keras.__version__,'train':len(a),'validacion':len(b),'test_humano':len(m),'train_grupos':len(set(g)),'validacion_grupos':d.iloc[b].listing_id.nunique(),'test_grupos':m.listing_id.nunique(),'grupos_compartidos':0,'svd_componentes':e['svd'],'criterio_seleccion':'Macro F1 automático; test humano no usado en ajustes de esta ejecución','python':platform.python_version(),'hardware':platform.machine(),'prevalencias_train':y.mean(axis=0).tolist()}
 (P/'datos/evidencia_sobreajuste.json').write_text(json.dumps(evidencia,indent=2)); (P/'datos/metodologia.json').write_text(json.dumps(evidencia,indent=2))
 # Curvas con subconjuntos anidados de entidades; validación fija.
 grupos=np.unique(g); np.random.default_rng(SEMILLA).shuffle(grupos); curvas=[]
 for frac in [.1,.25,.5,.75,1.]:
  idx=np.isin(g,grupos[:max(2,int(frac*len(grupos)))])
  for nombre in modelos:
   if frac==1: fitted=modelos[nombre]
   elif nombre=='MLP': fitted=ajustar_red(x.iloc[idx],y[idx],xv,yv,e)[0]
   else: fitted=clone(modelos[nombre]).fit(x.iloc[idx],y[idx])
   curvas.append({'modelo':nombre,'proporcion':frac,'grupos':len(set(g[idx])),'filas':int(idx.sum()),'Macro F1 automático':f1(yv,fitted.predict(xv))})
  print('CURVA',frac,flush=True)
 pd.DataFrame(curvas).to_csv(P/'datos/curva_aprendizaje.csv',index=False)
 from evaluar import finalizar
 finalizar(modelos,m,xv,yv,y,costos,busquedas,evidencia)
 print('RESULTADOS',pd.read_csv(P/'datos/metricas.csv').to_string(index=False),flush=True)
if __name__=='__main__': main()
