"""Evaluación final e interpretabilidad; sin reentrenar ni ajustar usando el test."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_recall_fscore_support,multilabel_confusion_matrix
from langdetect import detect,DetectorFactory,LangDetectException
from texto import TEMAS,SEMILLA,limpiar,palabras
P=Path(__file__).parent; DetectorFactory.seed=SEMILLA

def f1(y,p): return float(f1_score(y,p,average='macro',zero_division=0))
def idioma(t):
 try: return detect(t) if len(limpiar(t))>=20 else 'indeterminado'
 except LangDetectException: return 'indeterminado'
def etiquetas(v): return ', '.join(t for t,p in zip(TEMAS,v) if p) or 'ninguno'
def intervalo(y,p,g):
 rng=np.random.default_rng(SEMILLA); grupos=np.unique(g); bloques={v:np.flatnonzero(g==v) for v in grupos}; s=[]
 for _ in range(500):
  idx=np.concatenate([bloques[v] for v in rng.choice(grupos,len(grupos),replace=True)]); s.append(f1(y[idx],p[idx]))
 return np.quantile(s,[.025,.975]).tolist()
def finalizar(modelos,m,xv,yv,ytrain,costos,busquedas,e):
 xt=m[['comments']]; yt=m[TEMAS].to_numpy(int); g=m.listing_id.to_numpy(); pred={}; resultados=[]; detalles=[]; matrices=[]
 for nombre,mod in modelos.items():
  mod.predict(xt.iloc[:2]); tiempos=[]
  for _ in range(7):
   inicio=time.perf_counter(); pr=mod.predict(xt); tiempos.append(time.perf_counter()-inicio)
  pred[nombre]=pr; intervalo95=intervalo(yt,pr,g)
  size=sum(p.stat().st_size for p in (P/'modelos').glob('keras_*'))+(P/'modelos/red_keras.keras').stat().st_size if nombre=='MLP' else (P/(nombre+'.joblib')).stat().st_size
  info={'modelo':nombre,'Macro F1 manual':f1(yt,pr),'Micro F1 manual':f1_score(yt,pr,average='micro',zero_division=0),'Coincidencia completa':float(np.mean(np.all(yt==pr,axis=1))),'entrenamiento_s':costos[nombre],'inferencia_ms_resena':float(np.median(tiempos)*1000/len(xt)),'tamano_MB':size/1e6,'IC95_inferior':intervalo95[0],'IC95_superior':intervalo95[1],'F1 validación automática':f1(yv,mod.predict(xv))}
  if nombre!='MLP': info['F1 CV automático']=next(v['F1 CV automático'] for v in busquedas if v['modelo']==nombre)
  resultados.append(info)
  precis,rec,ff,support=precision_recall_fscore_support(yt,pr,average=None,zero_division=0)
  detalles.extend({'modelo':nombre,'tema':t,'precision':p,'recall':r,'F1':f,'positivos':int(n)} for t,p,r,f,n in zip(TEMAS,precis,rec,ff,support))
  for t,cm in zip(TEMAS,multilabel_confusion_matrix(yt,pr)):
   matrices.append({'modelo':nombre,'tema':t,'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1])})
 pd.DataFrame(resultados).to_csv(P/'datos/metricas.csv',index=False);pd.DataFrame(detalles).to_csv(P/'datos/metricas_tema.csv',index=False);pd.DataFrame(matrices).to_csv(P/'datos/confusion.csv',index=False)
 trivial=np.tile((ytrain.mean(axis=0)>=.5).astype(int),(len(yt),1))
 (P/'datos/baseline.json').write_text(json.dumps({'modelo':'Mayoritaria por etiqueta','etiquetas':trivial[0].tolist(),'Macro F1 manual':f1(yt,trivial),'Micro F1 manual':float(f1_score(yt,trivial,average='micro',zero_division=0)),'Coincidencia completa':float(np.mean(np.all(yt==trivial,axis=1)))},indent=2))
 # Comparación de errores por reseña, además de matrices por etiqueta.
 er={n:np.any(p!=yt,axis=1) for n,p in pred.items()};sol=[]
 for c in ['LogisticRegression','LinearSVC']:
  sol.append({'clasico':c,'ambos_fallan':int(np.sum(er[c]&er['MLP'])),'solo_clasico':int(np.sum(er[c]&~er['MLP'])),'solo_red':int(np.sum(~er[c]&er['MLP'])),'ambos_aciertan':int(np.sum(~er[c]&~er['MLP']))})
 pd.DataFrame(sol).to_csv(P/'datos/solapamiento_errores.csv',index=False)
 # Guardar textos únicamente en datos privados; la publicación utiliza extractos anonimizados.
 errors=[]
 for i in np.flatnonzero(er['LogisticRegression']|er['MLP'])[:5]:
  errors.append({'fila_manual':int(i),'reseña':m.iloc[i].comments,'manual':etiquetas(yt[i]),'LogisticRegression':etiquetas(pred['LogisticRegression'][i]),'LinearSVC':etiquetas(pred['LinearSVC'][i]),'MLP':etiquetas(pred['MLP'][i])})
 (P/'datos/errores_privados.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 lenguajes=m.comments.map(idioma).to_numpy(); sesgos=[]
 for lang in sorted(set(lenguajes)):
  idx=lenguajes==lang; activos=yt[idx].sum(axis=0)>0
  for n,pr in pred.items():
   sesgos.append({'idioma':lang,'modelo':n,'reseñas':int(idx.sum()),'grupos':int(len(set(g[idx]))),'temas_con_positivos':int(activos.sum()),'Macro F1':f1(yt[idx],pr[idx]) if idx.sum()>=5 and yt[idx].sum()>0 else None,'F1 temas presentes':f1(yt[idx][:,activos],pr[idx][:,activos]) if idx.sum()>=5 and activos.sum()>1 else None})
 pd.DataFrame(sesgos).to_csv(P/'datos/sesgos_idioma.csv',index=False)
 # Interpretación clásica: coeficientes lineales de las raíces TF-IDF.
 terms=[]
 for n in ['LogisticRegression','LinearSVC']:
  mod=modelos[n]; names=mod['preparacion'].named_transformers_['texto'].get_feature_names_out()
  for t,est in zip(TEMAS,mod['clasificador'].estimators_):
   co=est.coef_[0,:len(names)]
   for j in np.argsort(co)[-8:][::-1]: terms.append({'modelo':n,'tema':t,'termino':names[j],'coeficiente':float(co[j])})
 pd.DataFrame(terms).to_csv(P/'datos/terminos_clasicos.csv',index=False)
 # Permutation importance sobre entradas densas ya ajustadas, nunca refit.
 red=modelos['MLP']; z=red.preparacion.transform(xv.iloc[:600]).astype('float32'); real=yv[:600]
 def score(a): return f1(real,np.asarray(red.red(a,training=False))>=.5)
 base=score(z); rng=np.random.default_rng(SEMILLA); im=[]
 for j in range(z.shape[1]):
  drops=[]
  for _ in range(3):
   perm=z.copy();perm[:,j]=perm[rng.permutation(len(z)),j];drops.append(base-score(perm))
  im.append({'entrada':j,'variable':f'SVD {j+1}' if j<z.shape[1]-1 else 'Longitud','caida_Macro_F1':float(np.mean(drops)),'desviacion':float(np.std(drops))})
 pd.DataFrame(im).to_csv(P/'datos/importancia_red.csv',index=False)
 pre=red.preparacion['columnas'].named_transformers_['texto']; names=pre['tfidf'].get_feature_names_out(); cargas=[]
 for row in sorted(im,key=lambda v:v['caida_Macro_F1'],reverse=True)[:5]:
  j=row['entrada']
  if j<len(pre['svd'].components_):
   c=pre['svd'].components_[j];indices=np.argsort(np.abs(c))[-6:][::-1]
   cargas.append({'componente':j+1,'terminos_cargas_absolutas':', '.join(names[k] for k in indices),'caida_Macro_F1':row['caida_Macro_F1']})
 (P/'datos/cargas_svd.json').write_text(json.dumps(cargas,ensure_ascii=False,indent=2))
 # Pruebas de comportamiento separadas de métricas; no son datos de entrenamiento.
 textos=['hay mucho ruido','todo estaba muy limpio','el anfitrión respondió rápido','está cerca del metro','el precio es caro','limpio pero ruidoso','El apartamento era silencioso','La vista era preciosa','Costanera Center está cerca','The apartment was noisy','Great host and clean apartment']
 probes=[]
 for texto in textos:
  for n,mod in modelos.items():
   xx=pd.DataFrame({'comments':[texto]});pr=mod.predict(xx)[0];p=mod.predict_proba(xx)[0].tolist() if hasattr(mod,'predict_proba') else None
   probes.append({'texto':texto,'modelo':n,'prediccion':etiquetas(pr),'puntuaciones':p})
 (P/'datos/pruebas_comportamiento.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2))
 # Decisión fijada por criterio de costo y salida probabilística, sin usar test para ajustar.
 (P/'datos/decision.json').write_text(json.dumps({'modelo':'LogisticRegression','razon':'Salida probabilística no calibrada, coeficientes interpretables, modelo pequeño y entrenamiento rápido. La selección final considera los resultados humanos sin reajustar parámetros.','umbral':.5,'probabilidades_calibradas':False},ensure_ascii=False,indent=2))
 from generar_informe import generar
 generar()
