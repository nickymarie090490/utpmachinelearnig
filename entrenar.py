"""Entrenamiento por alojamiento. La muestra humana solo se usa al final."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
from texto import TEMAS, SEMILLA, preparar
P=Path(__file__).parent
if __name__=='__main__':
 d=pd.read_csv(P/'datos/entrenamiento_debil.csv.gz',dtype={'listing_id':str},keep_default_na=False)
 m=pd.read_csv(P/'datos/muestra_manual.csv',dtype={'listing_id':str},keep_default_na=False)
 assert not set(d.listing_id)&set(m.listing_id)
 a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEMILLA).split(d,groups=d.listing_id))
 x=d.iloc[a][['comments']]; y=d.iloc[a][TEMAS].to_numpy(int); g=d.iloc[a].listing_id
 cv=GroupKFold(3); scorer=make_scorer(f1_score,average='macro',zero_division=0)
 models={}; results=[]; detail=[]
 for name,est in [('LogisticRegression',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=1000,random_state=SEMILLA)),('LinearSVC',LinearSVC(class_weight='balanced',dual='auto',max_iter=3000,random_state=SEMILLA))]:
  start=time.perf_counter()
  pipe=Pipeline([('preparacion',preparar()),('clasificador',OneVsRestClassifier(est))])
  search=GridSearchCV(pipe,{'clasificador__estimator__C':[.5,2],'preparacion__texto__ngram_range':[(1,1),(1,2)]},scoring=scorer,cv=cv,n_jobs=1)
  search.fit(x,y,groups=g); models[name]=search.best_estimator_
  results.append({'modelo':name,'F1 CV automático':search.best_score_,'entrenamiento_s':time.perf_counter()-start})
  print('Entrenado',name,flush=True)
 # MLP con sigmoid multietiqueta. Validación externa por alojamiento;
 # sin early_stopping interno para evitar separar filas del mismo alojamiento.
 candidates=[]
 for layers in [(64,),(128,64)]:
  start=time.perf_counter()
  net=Pipeline([('preparacion',preparar(red=True)),('clasificador',MLPClassifier(hidden_layer_sizes=layers,max_iter=70,early_stopping=False,random_state=SEMILLA,batch_size=128))])
  with warnings.catch_warnings(record=True) as records:
   warnings.simplefilter('always',ConvergenceWarning);net.fit(x,y)
  score=f1_score(d.iloc[b][TEMAS],net.predict(d.iloc[b][['comments']]),average='macro',zero_division=0)
  candidates.append((score,net,layers,time.perf_counter()-start,bool(records)))
 best=max(candidates,key=lambda v:v[0]);models['MLP']=best[1]
 results.append({'modelo':'MLP','F1 validación automática':best[0],'entrenamiento_s':best[3]})
 arch=[{'capas':str(c[2]),'F1 validación automática':c[0],'advertencia_convergencia':c[4]} for c in candidates]
 pd.DataFrame(arch).to_csv(P/'datos/arquitecturas.csv',index=False)
 for r in results:
  pred=models[r['modelo']].predict(m[['comments']]);true=m[TEMAS].to_numpy(int)
  r.update({'Macro F1 manual':f1_score(true,pred,average='macro',zero_division=0),'Micro F1 manual':f1_score(true,pred,average='micro',zero_division=0),'Coincidencia completa':float(np.mean(np.all(pred==true,axis=1)))})
  pr,rec,f,_=precision_recall_fscore_support(true,pred,average=None,zero_division=0)
  detail.extend({'modelo':r['modelo'],'tema':t,'precision':p,'recall':q,'F1':z} for t,p,q,z in zip(TEMAS,pr,rec,f))
  joblib.dump(models[r['modelo']],P/(r['modelo']+'.joblib'))
 pd.DataFrame(results).to_csv(P/'datos/metricas.csv',index=False)
 pd.DataFrame(detail).to_csv(P/'datos/metricas_tema.csv',index=False)
 (P/'datos/metodologia.json').write_text(json.dumps({'train':len(a),'validacion':len(b),'test_humano':len(m),'MLP':'scikit-learn, distinta de la implementación TensorFlow del notebook','arquitectura':best[2]},indent=2))
 print(pd.DataFrame(results).to_string(),flush=True)
