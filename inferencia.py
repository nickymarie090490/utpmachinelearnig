"""Validación de entradas y explicación local de las predicciones."""
import re
import numpy as np
import pandas as pd
from texto import TEMAS,palabras

def validar_textos(textos):
 textos=list(textos)
 if not 0<len(textos)<=5000:raise ValueError('Se requieren entre 1 y 5.000 reseñas.')
 if any(not isinstance(t,str) or not t.strip() or not any(c.isalpha() for c in t) for t in textos):raise ValueError('Cada reseña debe contener texto y al menos una letra.')
 if any(len(t)>15000 for t in textos):raise ValueError('Una reseña supera los 15.000 caracteres.')
 return pd.DataFrame({'comments':textos})
def analizar_modelo(modelo,textos):
 x=validar_textos(textos);scores=modelo.predict_proba(x) if hasattr(modelo,'predict_proba') else None
 pred=(scores>=.5).astype(int) if scores is not None else modelo.predict(x)
 if pred.shape!=(len(x),5) or not np.isin(pred,[0,1]).all():raise ValueError('Salida del modelo inválida.')
 if scores is not None and (not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):raise ValueError('Puntuaciones inválidas.')
 return pred,scores

def explicar_local(modelo,texto,nombre):
 x=validar_textos([texto]);rows=[]
 if nombre!='MLP':
  pre=modelo['preparacion'];z=pre.transform(x);names=list(pre.named_transformers_['texto'].get_feature_names_out())+['longitud']
  values=z.toarray()[0] if hasattr(z,'toarray') else z[0]
  for t,est in zip(TEMAS,modelo['clasificador'].estimators_):
   contrib=values*est.coef_[0]
   for j in np.argsort(np.abs(contrib))[-3:][::-1]:
    if abs(contrib[j])>1e-8:rows.append({'Tema':t,'Evidencia':names[j],'Efecto':float(contrib[j]),'Método':'Contribución a la puntuación lineal'})
 else:
  base=modelo.predict_proba(x)[0];tokens=list(dict.fromkeys(re.findall(r'\b\w+\b',texto)))
  tokens=[t for t in tokens if palabras(t)][:15]
  altered=[re.sub(r'\b'+re.escape(t)+r'\b',' ',texto,flags=re.I) for t in tokens]
  if altered:
   out=modelo.predict_proba(pd.DataFrame({'comments':altered}))
   for j,t in enumerate(TEMAS):
    delta=base[j]-out[:,j]
    for k in np.argsort(np.abs(delta))[-2:][::-1]:rows.append({'Tema':t,'Evidencia':tokens[k],'Efecto':float(delta[k]),'Método':'Cambio de puntuación al quitar el término'})
 return pd.DataFrame(rows)
