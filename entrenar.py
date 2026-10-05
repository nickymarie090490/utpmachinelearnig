"""Problema 2: clásicos y MLP Keras. No se utilizan etiquetas humanas para ajustar modelos.
python entrenar.py: recalcula todo. python entrenar.py --solo-red: conserva los clásicos.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_recall_fscore_support
from texto import TEMAS, SEMILLA, preparar
from keras_red import RedKeras

P = Path(__file__).parent
MAX_EPOCAS = 100
PACIENCIA = 6
MIN_DELTA = .0001
BATCH = 128
TF_VERSION = tf.__version__
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()

def lotes(x, y, entrenamiento=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y.astype('float32')))
    if entrenamiento:
        ds = ds.shuffle(len(x), seed=SEMILLA, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return ds.batch(BATCH).with_options(options)

def crear_red(n, capas, dropout):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    red = tf.keras.Sequential([tf.keras.layers.Input(shape=(n,), name='texto_denso')], name='mlp_resenas')
    for i, unidades in enumerate(capas):
        red.add(tf.keras.layers.Dense(unidades, activation='relu', name=f'oculta_{i+1}'))
        if dropout:
            red.add(tf.keras.layers.Dropout(dropout, name=f'dropout_{i+1}'))
    red.add(tf.keras.layers.Dense(len(TEMAS), activation='sigmoid', name='cinco_temas'))
    red.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='binary_crossentropy')
    return red

def bce(y, probs):
    probs = np.clip(probs, 1e-7, 1-1e-7)
    return float(np.mean(-(y*np.log(probs)+(1-y)*np.log(1-probs))))

def exportar(pre, red):
    p = P/'modelos'; p.mkdir(exist_ok=True)
    buffer = io.BytesIO(); joblib.dump(pre, buffer, compress=3); raw = buffer.getvalue()
    parts = []
    for old in p.glob('keras_pre.part*'):
        old.unlink()
    for i, start in enumerate(range(0, len(raw), 500000)):
        name = f'keras_pre.part{i:02d}'; block = raw[start:start+500000]
        (p/name).write_bytes(block)
        parts.append({'archivo':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
    red.save(p/'red_keras.keras')
    (p/'keras_manifest.json').write_text(json.dumps({'tensorflow':TF_VERSION,'preparacion_parts':parts,'preparacion_sha256':hashlib.sha256(raw).hexdigest(),'red_sha256':hashlib.sha256((p/'red_keras.keras').read_bytes()).hexdigest()}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--solo-red', action='store_true'); args = parser.parse_args()
    d = pd.read_csv(P/'datos/entrenamiento_debil.csv.gz',dtype={'listing_id':str},keep_default_na=False)
    m = pd.read_csv(P/'datos/muestra_manual.csv',dtype={'listing_id':str},keep_default_na=False)
    assert not set(d.listing_id)&set(m.listing_id)
    a, b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEMILLA).split(d,groups=d.listing_id))
    assert not set(d.iloc[a].listing_id)&set(d.iloc[b].listing_id)
    x = d.iloc[a][['comments']]; y = d.iloc[a][TEMAS].to_numpy(int); g = d.iloc[a].listing_id
    xv = d.iloc[b][['comments']]; yv = d.iloc[b][TEMAS].to_numpy(int)
    models = {}; results = []; detail = []
    for name, est in [('LogisticRegression',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=1000,random_state=SEMILLA)),('LinearSVC',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=SEMILLA))]:
        if args.solo_red:
            models[name] = joblib.load(P/(name+'.joblib'))
            old = pd.read_csv(P/'datos/metricas.csv'); results.append(old[old.modelo==name].iloc[0].dropna().to_dict())
        else:
            start = time.perf_counter()
            pipe = Pipeline([('preparacion',preparar()),('clasificador',OneVsRestClassifier(est))])
            search = GridSearchCV(pipe,{'clasificador__estimator__C':[.5,2],'preparacion__texto__ngram_range':[(1,1),(1,2)]},scoring=make_scorer(f1_score,average='macro',zero_division=0),cv=GroupKFold(3),n_jobs=1)
            search.fit(x,y,groups=g); models[name] = search.best_estimator_
            results.append({'modelo':name,'F1 CV automático':search.best_score_,'entrenamiento_s':time.perf_counter()-start})
            joblib.dump(models[name],P/(name+'.joblib'))
    start = time.perf_counter()
    pre = preparar(red=True)
    z = pre.fit_transform(x).astype('float32'); zv = pre.transform(xv).astype('float32')
    train_ds = lotes(z,y,True); val_ds = lotes(zv,yv)
    candidatos = []; histories = []
    for capas, dropout in [((64,),.2),((128,64),.3)]:
        red = crear_red(z.shape[1],capas,dropout)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=PACIENCIA,min_delta=MIN_DELTA,restore_best_weights=True,verbose=0)
        h = red.fit(train_ds,validation_data=val_ds,epochs=MAX_EPOCAS,callbacks=[callback],shuffle=False,verbose=0).history
        pred = (np.asarray(red(zv,training=False))>=.5).astype(int)
        score = f1_score(yv,pred,average='macro',zero_division=0)
        candidato = {'capas':str(capas),'dropout':dropout,'F1 validación automática':float(score),'epocas_ejecutadas':len(h['loss']),'mejor_epoca':int(callback.best_epoch+1),'early_stopping_activo':True,'parada_anticipada':bool(callback.stopped_epoch>0),'val_loss_restaurada':bce(yv,np.asarray(red(zv,training=False)))}
        candidatos.append((score,red,capas,dropout,h,candidato)); histories.append(candidato)
        print('Candidato Keras',candidato,flush=True)
    best = max(candidatos,key=lambda c:c[0])
    _, red, capas, dropout, h, elegido = best
    models['MLP'] = RedKeras(pre,red)
    results.append({'modelo':'MLP','F1 validación automática':best[0],'entrenamiento_s':time.perf_counter()-start})
    pd.DataFrame(histories).to_csv(P/'datos/arquitecturas.csv',index=False)
    pd.DataFrame({'epoca':range(1,len(h['loss'])+1),'Entrenamiento':h['loss'],'Validación':h['val_loss']}).to_csv(P/'datos/perdida_keras.csv',index=False)
    # Ablación exploratoria: misma arquitectura, representación, división y semilla.
    # Desactiva ambos controles para observar sobreajuste. Nunca se evalúa contra el test humano.
    baseline = crear_red(z.shape[1],capas,0.)
    hb = baseline.fit(lotes(z,y,True),validation_data=val_ds,epochs=MAX_EPOCAS,shuffle=False,verbose=0).history
    pd.DataFrame({'epoca':range(1,len(hb['loss'])+1),'Entrenamiento':hb['loss'],'Validación':hb['val_loss']}).to_csv(P/'datos/perdida_sin_control.csv',index=False)
    prob_train = models['MLP'].predict_proba(x); prob_val = models['MLP'].predict_proba(xv)
    bt = np.asarray(baseline(z,training=False)); bv = np.asarray(baseline(zv,training=False))
    evidence = {'loss_train_restaurada':bce(y,prob_train),'loss_val_restaurada':bce(yv,prob_val),'loss_val_ultima_epoca_regularizada':float(h['val_loss'][-1]),'loss_train_sin_control':bce(y,bt),'loss_val_sin_control':bce(yv,bv),'F1_val_sin_control':float(f1_score(yv,(bv>=.5).astype(int),average='macro',zero_division=0)),'mejor_epoca':elegido['mejor_epoca'],'epocas_ejecutadas':len(h['loss']),'parada_anticipada':elegido['parada_anticipada'],'max_epocas':MAX_EPOCAS,'patience':PACIENCIA,'min_delta':MIN_DELTA,'arquitectura_oculta':list(capas),'dropout':dropout,'entrada_dimensiones':int(z.shape[1]),'capas_densas':len(capas)+1,'salida_unidades':5,'activacion_oculta':'relu','activacion_salida':'sigmoid','optimizador':'Adam','learning_rate':.001,'loss':'binary_crossentropy','batch_size':BATCH,'tensorflow':TF_VERSION,'keras':tf.keras.__version__,'criterio_seleccion':'Macro F1 en validación automática; parada por val_loss','train':len(a),'validacion':len(b),'test_humano':len(m),'train_grupos':int(d.iloc[a].listing_id.nunique()),'validacion_grupos':int(d.iloc[b].listing_id.nunique()),'grupos_compartidos':0,'svd_componentes':64,'svd_varianza_explicada':float(pre['columnas'].named_transformers_['texto']['svd'].explained_variance_ratio_.sum()),'representacion':'TF-IDF → TruncatedSVD(64) + log-longitud escalada → StandardScaler (65 entradas)','ablacion':'Misma arquitectura y partición; sin dropout ni early stopping, 100 épocas. Una semilla: evidencia exploratoria, no prueba causal de cada control.'}
    (P/'datos/evidencia_sobreajuste.json').write_text(json.dumps(evidence,indent=2))
    for r in results:
        pred = models[r['modelo']].predict(m[['comments']]); true = m[TEMAS].to_numpy(int)
        r.update({'Macro F1 manual':f1_score(true,pred,average='macro',zero_division=0),'Micro F1 manual':f1_score(true,pred,average='micro',zero_division=0),'Coincidencia completa':float(np.mean(np.all(pred==true,axis=1)))})
        pr,rec,f,_ = precision_recall_fscore_support(true,pred,average=None,zero_division=0)
        detail.extend({'modelo':r['modelo'],'tema':t,'precision':p,'recall':q,'F1':v} for t,p,q,v in zip(TEMAS,pr,rec,f))
    pd.DataFrame(results).to_csv(P/'datos/metricas.csv',index=False)
    pd.DataFrame(detail).to_csv(P/'datos/metricas_tema.csv',index=False)
    (P/'datos/metodologia.json').write_text(json.dumps(evidence,indent=2))
    exportar(pre,red)
    print(pd.DataFrame(results).to_string(),flush=True)
    print('EVIDENCIA',json.dumps(evidence),flush=True)
    from generar_informe import generar
    generar()
