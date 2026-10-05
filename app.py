from pathlib import Path
import json
import io
import hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from texto import TEMAS
P=Path(__file__).parent
st.set_page_config(page_title='Temas de reseñas · Equipo 6',page_icon='💬',layout='wide')
st.title('💬 ¿De qué habla esta reseña?')
st.write('Detecta limpieza, ruido, ubicación, anfitrión y precio. Una reseña puede mencionar varios temas.')
@st.cache_resource
def cargar(nombre):
 if nombre == 'MLP' and (P/'modelos/MLP.sha256').exists():
  raw=b''.join(f.read_bytes() for f in sorted((P/'modelos').glob('MLP.part*')))
  expected=(P/'modelos/MLP.sha256').read_text().strip()
  if hashlib.sha256(raw).hexdigest()!=expected:
   raise ValueError('El archivo de la red neuronal está incompleto. Revisa los archivos modelos/MLP.part*.')
  return joblib.load(io.BytesIO(raw))
 return joblib.load(P/(nombre+'.joblib'))
@st.cache_data
def tabla(nombre): return pd.read_csv(P/'datos'/nombre,keep_default_na=False)
available=[name for name in ['LogisticRegression','LinearSVC','MLP'] if (P/(name+'.joblib')).exists() or (name=='MLP' and (P/'modelos/MLP.sha256').exists() and len(list((P/'modelos').glob('MLP.part*')))==10)]
if not available:
 st.error('No hay modelos disponibles en este despliegue.'); st.stop()
model_name=st.sidebar.selectbox('Modelo',available)
if 'MLP' not in available:
 st.sidebar.info('La red neuronal está temporalmente fuera de servicio. Puedes utilizar los dos modelos clásicos.')
st.sidebar.caption('MLP es una red neuronal. Los otros dos son modelos clásicos.')
if not (P/(model_name+'.joblib')).exists() and not (model_name == 'MLP' and (P/'modelos/MLP.sha256').exists()):
 st.error('Faltan los modelos. Ejecuta python entrenar.py dentro de la carpeta del proyecto.');st.stop()
model=cargar(model_name)
t1,t2,t3,t4=st.tabs(['Una reseña','Varias reseñas','Resultados','Cómo funciona'])
with t1:
 with st.form('resena'):
  text=st.text_area('Escribe una reseña',value='El apartamento estaba limpio y cerca del metro, pero había mucho ruido.',height=140,max_chars=15000)
  submit=st.form_submit_button('Analizar reseña',type='primary')
 if submit:
  if not text.strip(): st.warning('Escribe una reseña para analizar.')
  else:
   x=pd.DataFrame({'comments':[text]}); pred=model.predict(x)[0]
   topics=[t for t,p in zip(TEMAS,pred) if p]
   if topics:
    st.success('Temas detectados: '+', '.join(topics))
   else:
    st.info('No se detectaron temas con el umbral del modelo.')
   out=pd.DataFrame({'Tema':TEMAS,'Detectado':['Sí' if p else 'No' for p in pred]})
   if hasattr(model,'predict_proba'):
    probs=model.predict_proba(x)[0];out['Puntuación (0–1)']=probs
    st.bar_chart(pd.DataFrame({'Puntuación':probs},index=TEMAS))
    st.caption('Puntuaciones sin calibrar; no representan una certeza comprobada. Umbral fijo: 0.5.')
   st.dataframe(out,hide_index=True)
 st.caption('Detectar un tema no significa que la opinión sea negativa. «Muy limpio» también menciona limpieza.')
with t2:
 st.write('Sube un CSV UTF-8 con una columna comments. Máximo 5 000 reseñas por archivo.')
 file=st.file_uploader('Archivo de reseñas',type=['csv'])
 if file and st.button('Analizar archivo'):
  try:
   d=pd.read_csv(file,keep_default_na=False)
   if 'comments' not in d: raise ValueError('El CSV debe incluir la columna comments.')
   if not 0<len(d)<=5000: raise ValueError('El archivo debe contener entre 1 y 5 000 filas.')
   if d.comments.astype(str).str.len().max()>15000: raise ValueError('Una reseña supera 15 000 caracteres.')
   d['comments']=d.comments.astype(str); pred=model.predict(d[['comments']])
   output=pd.concat([d.reset_index(drop=True),pd.DataFrame(pred,columns=['pred_'+t for t in TEMAS])],axis=1)
   st.dataframe(output,hide_index=True)
   st.bar_chart(pd.DataFrame({'Reseñas que mencionan el tema':pred.sum(axis=0)},index=TEMAS))
   st.download_button('Descargar resultados CSV',output.to_csv(index=False).encode('utf-8-sig'),'predicciones.csv','text/csv')
  except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as e: st.error(str(e))
 st.download_button('Descargar ejemplo CSV','comments\n"Limpio y cerca del metro, pero ruidoso."\n'.encode(),'ejemplo.csv','text/csv')
with t3:
 metrics=tabla('metricas.csv')
 st.subheader('Evaluación con etiquetas humanas')
 st.dataframe(metrics,hide_index=True)
 st.bar_chart(metrics.set_index('modelo')[['Macro F1 manual','Micro F1 manual']])
 st.dataframe(tabla('metricas_tema.csv').query('modelo == @model_name'),hide_index=True)
 st.subheader('Arquitecturas de la red neuronal')
 st.dataframe(tabla('arquitecturas.csv'),hide_index=True)
 st.caption('Macro F1 da el mismo peso a cada tema. La muestra manual es pequeña y dirigida; estos resultados no garantizan el desempeño en toda la población.')
with t4:
 source=json.loads((P/'datos/origen.json').read_text());method=json.loads((P/'datos/metodologia.json').read_text())
 st.write(f"Fuente: {source['originales']:,} reseñas del Excel. Etiquetas humanas completas: {source['manuales']}. Muestra automática: {source['entrenamiento']:,}.")
 st.write('1. Se excluyen del entrenamiento todos los alojamientos del conjunto humano y las copias de sus textos. Se eliminan textos normalizados repetidos y sin letras.')
 st.write('2. Se crean etiquetas débiles con reglas de palabras; pueden equivocarse. TF-IDF transforma los textos en números y se agrega su longitud.')
 st.write('3. Los modelos clásicos se seleccionan con GroupKFold de tres pliegues. La red usa 64 componentes SVD y dos arquitecturas candidatas, seleccionadas en una validación por alojamiento.')
 st.write('4. La muestra humana se utiliza al final para evaluar; no se ajustan umbrales con ella.')
 st.info('La red de esta aplicación usa MLPClassifier de scikit-learn. Es una red neuronal real, pero su implementación y sus resultados difieren de la red TensorFlow del notebook. No utiliza dropout ni early stopping.')
 st.json(method)
