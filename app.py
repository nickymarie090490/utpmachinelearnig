from pathlib import Path
import json
import io
import hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from texto import TEMAS

P = Path(__file__).parent
NOMBRES = {'limpieza': 'Limpieza', 'ruido': 'Ruido', 'ubicacion': 'Ubicación', 'anfitrion': 'Anfitrión', 'precio': 'Precio'}
MODELOS = {'LogisticRegression': 'Regresión logística', 'LinearSVC': 'SVM lineal', 'MLP': 'Red neuronal (MLP)'}
ICONOS = {'limpieza': '🧹', 'ruido': '🔊', 'ubicacion': '📍', 'anfitrion': '🤝', 'precio': '💰'}
st.set_page_config(page_title='Temas de reseñas · Equipo 6', page_icon='💬', layout='wide')
st.markdown('''<style>
.block-container {max-width:1180px;padding-top:2.5rem;padding-bottom:3rem;}
h1 {letter-spacing:-.035em;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
[data-testid="stSidebar"] h3 {font-size:1.1rem;}
[data-testid="stTabs"] button {font-size:.95rem;}
</style>''', unsafe_allow_html=True)
st.caption('EQUIPO 6 · ANÁLISIS DE RESEÑAS')
st.title('Entiende de qué hablan tus huéspedes')
st.write('Identifica cinco temas en una reseña o en un archivo. Obtén una lectura clara de cada resultado y compara los modelos.')

@st.cache_resource
def cargar(nombre):
    if nombre == 'MLP' and (P/'modelos/MLP.sha256').exists():
        raw = b''.join(f.read_bytes() for f in sorted((P/'modelos').glob('MLP.part*')))
        if hashlib.sha256(raw).hexdigest() != (P/'modelos/MLP.sha256').read_text().strip():
            raise ValueError('El archivo de la red neuronal está incompleto.')
        return joblib.load(io.BytesIO(raw))
    return joblib.load(P/(nombre+'.joblib'))

@st.cache_data
def tabla(nombre):
    return pd.read_csv(P/'datos'/nombre, keep_default_na=False)

@st.cache_data
def analizar(nombre, textos):
    modelo = cargar(nombre)
    x = pd.DataFrame({'comments': list(textos)})
    pred = modelo.predict(x)
    scores = modelo.predict_proba(x) if hasattr(modelo, 'predict_proba') else None
    return pred, scores

available = [n for n in MODELOS if (P/(n+'.joblib')).exists() or (n == 'MLP' and (P/'modelos/MLP.sha256').exists() and len(list((P/'modelos').glob('MLP.part*'))) == 10)]
if not available:
    st.error('No hay modelos disponibles.'); st.stop()
with st.sidebar:
    st.subheader('Elige cómo analizar')
    model_name = st.selectbox('Modelo', available, format_func=lambda n: MODELOS[n])
    if model_name == 'LogisticRegression':
        st.caption('Modelo clásico con puntuaciones por tema. Útil para explorar cómo se toma cada decisión.')
    elif model_name == 'LinearSVC':
        st.caption('Modelo clásico con el mejor Macro F1 en la muestra humana de este proyecto. Muestra detecciones, sin probabilidades.')
    else:
        st.caption('Red neuronal multietiqueta. Sus resultados pueden diferir de los modelos clásicos.')
    st.divider()
    st.markdown('**Empieza aquí**\n\n1. Escribe una reseña.\n2. Pulsa **Analizar reseña**.\n3. Lee los temas detectados.\n\nCambia de modelo para comparar la última reseña analizada.')
    st.caption('¿Primera vez? Abre la pestaña **Guía de uso**.')
    if 'MLP' not in available:
        st.info('La red neuronal está temporalmente fuera de servicio.')

t1, t2, t3, t4, t5 = st.tabs(['💬 Una reseña', '📂 Varias reseñas', '📊 Comparar modelos', '📖 Guía de uso', '🧠 Cómo funciona'])
with t1:
    st.subheader('Analiza una reseña')
    st.caption('Puedes escribir en español o pegar el texto original. El modelo puede detectar más de un tema.')
    with st.form('resena'):
        text = st.text_area('Escribe una reseña', value='El apartamento estaba limpio y cerca del metro, pero había mucho ruido.', height=125, max_chars=15000)
        submit = st.form_submit_button('Analizar reseña', type='primary')
    if submit:
        if text.strip():
            st.session_state['ultima_resena'] = text.strip()
        else:
            st.session_state.pop('ultima_resena', None)
            st.warning('Escribe una reseña antes de analizar.')
    if 'ultima_resena' in st.session_state:
        reviewed = st.session_state['ultima_resena']
        pred, scores = analizar(model_name, (reviewed,))
        pred = pred[0]
        probs = scores[0] if scores is not None else None
        detected = [NOMBRES[t] for t, val in zip(TEMAS, pred) if val]
        st.divider()
        st.subheader('Tu resultado')
        st.caption('Modelo: '+MODELOS[model_name]+' · Reseña analizada: «'+reviewed+'»')
        if detected:
            st.success('**'+str(len(detected))+(' tema detectado:** ' if len(detected)==1 else ' temas detectados:** ')+', '.join(detected)+'.')
        else:
            st.info('No se detectó ninguno de los cinco temas. Esto no significa que la reseña esté vacía o sea incorrecta.')
        columns = st.columns(5)
        for j, (tema, val) in enumerate(zip(TEMAS, pred)):
            with columns[j], st.container(border=True):
                st.markdown(ICONOS[tema])
                st.markdown('**'+NOMBRES[tema]+'**')
                if val:
                    st.markdown('**● Detectado**')
                else:
                    st.caption('○ No detectado')
                if probs is not None:
                    st.progress(float(np.clip(probs[j], 0, 1)))
                    score_label = '<0,1%' if probs[j] < .001 else '>99,9%' if probs[j] > .999 else f'{probs[j]*100:.1f}%'.replace('.', ',')
                    st.caption('Puntuación: '+score_label)
        st.markdown('**Cómo leerlo:** “Detectado” indica que el modelo considera que la reseña menciona ese tema. Puede ser un elogio, una crítica o una descripción.')
        if probs is not None:
            st.caption('La puntuación refleja la salida del modelo. Desde 50% se marca “Detectado”. No es una certeza comprobada ni la precisión del modelo; las puntuaciones no están calibradas.')
        else:
            st.caption('SVM presenta una decisión por tema. No calculamos porcentajes de confianza para este modelo.')
        with st.expander('Ver tabla y puntuaciones exactas'):
            out = pd.DataFrame({'Tema': [NOMBRES[t] for t in TEMAS], 'Resultado': ['Detectado' if p else 'No detectado' for p in pred]})
            if probs is not None:
                out['Puntuación del modelo'] = probs*100
            st.dataframe(out, hide_index=True, width='stretch', column_config={'Puntuación del modelo': st.column_config.NumberColumn(format='%.2f%%')})
        st.caption('Si editas el texto, pulsa Analizar reseña para actualizarlo. Cambiar de modelo actualiza este resultado sobre la última reseña analizada.')
    else:
        with st.container(border=True):
            st.markdown('**Prueba rápida**')
            st.write('Escribe “hay mucho ruido” y pulsa Analizar reseña. Después prueba “todo estaba muy limpio” para explorar otro tema.')

with t2:
    st.subheader('Analiza varias reseñas de una vez')
    st.write('Prepara un CSV UTF-8 con la columna **comments** y una reseña por fila. Puedes analizar hasta 5.000 reseñas; cada texto admite hasta 15.000 caracteres.')
    st.download_button('Descargar CSV de ejemplo', 'comments\n"Limpio y cerca del metro, pero ruidoso."\n"El anfitrión respondió rápido."\n'.encode('utf-8-sig'), 'ejemplo.csv', 'text/csv')
    file = st.file_uploader('Sube tu archivo CSV', type=['csv'])
    if file and st.button('Analizar archivo', type='primary'):
        try:
            d = pd.read_csv(file, keep_default_na=False)
            if 'comments' not in d:
                raise ValueError('Falta la columna comments. Descarga el ejemplo para ver el formato.')
            if not 0 < len(d) <= 5000:
                raise ValueError('El archivo debe contener entre 1 y 5.000 filas.')
            if d.comments.astype(str).str.len().max() > 15000:
                raise ValueError('Una reseña supera los 15.000 caracteres.')
            if d.comments.astype(str).str.strip().eq('').any():
                raise ValueError('Hay reseñas vacías. Completa o elimina esas filas antes de subir el archivo.')
            st.session_state['archivo_resenas'] = d
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
            st.session_state.pop('archivo_resenas', None)
            st.error(str(e))
    if 'archivo_resenas' in st.session_state:
        d = st.session_state['archivo_resenas']
        pred, _ = analizar(model_name, tuple(d.comments.astype(str)))
        n = len(d)
        c1, c2, c3 = st.columns(3)
        c1.metric('Reseñas analizadas', f'{n:,}')
        c2.metric('Con al menos un tema', f'{np.any(pred, axis=1).sum():,}')
        c3.metric('Sin temas detectados', f'{(~np.any(pred, axis=1)).sum():,}')
        st.caption('Modelo: '+MODELOS[model_name]+'. Cambiar de modelo actualiza el análisis del último archivo procesado.')
        summary = pd.DataFrame({'Tema': [NOMBRES[t] for t in TEMAS], 'Reseñas': pred.sum(axis=0)})
        summary['Porcentaje de reseñas'] = summary.Reseñas/n*100
        st.subheader('¿Qué temas aparecen más?')
        st.bar_chart(summary.set_index('Tema')[['Reseñas']], horizontal=True, color='#0d9488')
        st.dataframe(summary.sort_values('Reseñas', ascending=False), hide_index=True, width='stretch', column_config={'Porcentaje de reseñas': st.column_config.NumberColumn(format='%.1f%%')})
        st.caption('Una reseña puede mencionar varios temas; los porcentajes pueden sumar más de 100%. Estas cifras cuentan menciones, no quejas.')
        human = pd.DataFrame({'Reseña': d.comments.astype(str).tolist(), 'Temas detectados': [', '.join(NOMBRES[t] for t, v in zip(TEMAS, row) if v) or 'Ninguno' for row in pred]})
        with st.expander('Ver resultados por reseña', expanded=True):
            st.dataframe(human, hide_index=True, width='stretch')
        output = d.reset_index(drop=True).copy()
        output['temas_detectados'] = human['Temas detectados']
        for j, tema in enumerate(TEMAS):
            output['pred_'+tema] = pred[:, j]
        st.download_button('Descargar resultados CSV', output.to_csv(index=False).encode('utf-8-sig'), 'predicciones.csv', 'text/csv')
        st.caption('En el CSV descargado, 1 significa detectado y 0 significa no detectado.')

with t3:
    metrics = tabla('metricas.csv')
    st.subheader('¿Qué tan bien funcionó cada modelo?')
    st.write('Comparamos las predicciones con las etiquetas humanas de 100 reseñas reservadas para evaluación. Son resultados del proyecto, no una garantía para cualquier texto nuevo.')
    cols = st.columns(len(metrics))
    for col, (_, row) in zip(cols, metrics.iterrows()):
        with col, st.container(border=True):
            st.markdown('**'+MODELOS[row['modelo']]+'**')
            st.metric('Macro F1', f"{float(row['Macro F1 manual']):.3f}")
            st.caption(f"Todas las etiquetas correctas en {float(row['Coincidencia completa']):.0%} de las reseñas evaluadas.")
    st.info('Macro F1 combina la capacidad de detectar temas reales y evitar detecciones incorrectas, dando el mismo peso a los cinco temas. Va de 0 a 1; cuanto más alto, mejor. Un F1 de 0,926 no significa 92,6% de reseñas completamente correctas.')
    comparison = metrics[['modelo', 'Macro F1 manual', 'Micro F1 manual']].copy()
    comparison['Modelo'] = comparison.modelo.map(MODELOS)
    st.bar_chart(comparison.set_index('Modelo')[['Macro F1 manual', 'Micro F1 manual']], horizontal=True)
    st.subheader('Desempeño por tema · '+MODELOS[model_name])
    detail = tabla('metricas_tema.csv')
    detail = detail[detail.modelo == model_name].copy()
    detail['Tema'] = detail.tema.map(NOMBRES)
    detail = detail.rename(columns={'precision': 'Precisión', 'recall': 'Cobertura', 'F1': 'F1 por tema'})
    st.dataframe(detail[['Tema', 'Precisión', 'Cobertura', 'F1 por tema']], hide_index=True, width='stretch', column_config={c: st.column_config.NumberColumn(format='%.3f') for c in ['Precisión', 'Cobertura', 'F1 por tema']})
    with st.expander('Entender las métricas'):
        st.markdown('**Precisión:** de las veces que el modelo marcó un tema, cuántas fueron correctas.\n\n**Cobertura (recall):** de las menciones reales de un tema, cuántas encontró.\n\n**F1:** combina precisión y cobertura.\n\n**Micro F1:** reúne las decisiones de todos los temas; los temas más frecuentes influyen más.\n\n**Todas las etiquetas correctas:** la predicción completa coincide con las cinco etiquetas humanas de una reseña.')
    st.caption('La muestra humana es pequeña y dirigida. No representa necesariamente todas las reseñas ni todos los idiomas.')
    with st.expander('Ver detalles del entrenamiento de la red'):
        st.dataframe(tabla('arquitecturas.csv'), hide_index=True, width='stretch')
        st.write('Se probaron capas (64) y (128, 64). La elección se hizo con etiquetas automáticas, antes de evaluar con etiquetas humanas. Ambas alcanzaron el límite de iteraciones; la convergencia no está garantizada.')

with t4:
    st.subheader('Guía de uso')
    st.write('Sigue estos pasos para empezar y entender lo que estás viendo.')
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown('### 1 · Analiza un texto')
        st.markdown('Abre **Una reseña**, escribe o pega el comentario y pulsa **Analizar reseña**.\n\nLee primero el resumen. Luego revisa las cinco tarjetas: **Detectado** o **No detectado**.')
    with b, st.container(border=True):
        st.markdown('### 2 · Interpreta el resultado')
        st.markdown('Cada tarjeta representa un tema. Una reseña puede mencionar varios.\n\n“Hay mucho ruido” menciona ruido; “es muy silencioso” también habla del tema ruido. La app identifica temas, no el sentimiento.')
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown('### 3 · Compara modelos')
        st.markdown('Cambia el modelo en el panel izquierdo. Se vuelve a analizar la última reseña que enviaste.\n\nAbre **Comparar modelos** para consultar su desempeño en las 100 reseñas humanas.')
    with b, st.container(border=True):
        st.markdown('### 4 · Procesa un archivo')
        st.markdown('Abre **Varias reseñas** y descarga el CSV de ejemplo. Conserva el encabezado **comments**, coloca una reseña por fila y guarda en UTF-8.\n\nSube el archivo, pulsa **Analizar archivo** y descarga los resultados.')
    st.subheader('Los cinco temas')
    st.table(pd.DataFrame({'Tema': ['Limpieza', 'Ruido', 'Ubicación', 'Anfitrión', 'Precio'], 'Qué puede mencionar': ['Aseo, suciedad, polvo, olores o higiene.', 'Ruido, silencio, sonidos o aislamiento acústico.', 'Zona, cercanía, metro, transporte o localización.', 'Atención, comunicación, trato o respuestas.', 'Costo, tarifas, valor, descuentos o reembolsos.']}))
    with st.expander('¿Qué significa una puntuación de 80%?', expanded=True):
        st.write('Es la puntuación del modelo para ese tema. En regresión logística y la red neuronal, desde 50% se marca detectado. No significa que la predicción tenga una exactitud comprobada del 80%. SVM muestra decisiones sin estos porcentajes.')
    with st.expander('¿Por qué los modelos dan resultados diferentes?'):
        st.write('Aprenden de los mismos tipos de datos con métodos distintos. Pueden equivocarse, especialmente con textos cortos, términos nuevos o idiomas poco representados. Conservamos estas diferencias para comparar su desempeño real.')
    with st.expander('¿Qué hago si no se detecta ningún tema?'):
        st.write('Revisa si la reseña habla de alguno de los cinco temas. Puede tratar otro asunto o el modelo puede haberlo omitido. Prueba otro modelo para comparar y revisa el texto original; no se debe asumir que el resultado es siempre correcto.')

with t5:
    source = json.loads((P/'datos/origen.json').read_text())
    method = json.loads((P/'datos/metodologia.json').read_text())
    st.subheader('De las reseñas a la predicción')
    c1, c2, c3 = st.columns(3)
    c1.metric('Reseñas en el Excel', f"{source['originales']:,}")
    c2.metric('Muestra de entrenamiento automático', f"{source['entrenamiento']:,}")
    c3.metric('Reseñas con etiquetas humanas', str(source['manuales']))
    st.markdown('**1. Preparar los datos.** Se eliminan textos repetidos y sin letras. Los alojamientos del conjunto humano y las copias de sus textos se excluyen del entrenamiento.\n\n**2. Enseñar a los modelos.** Reglas de palabras crean etiquetas automáticas imperfectas. TF-IDF convierte los textos en números y se añade su longitud.\n\n**3. Comparar alternativas.** Los modelos clásicos usan validación de tres pliegues por alojamiento. La red reduce el texto a 64 componentes SVD y compara dos arquitecturas con una validación separada por alojamiento.\n\n**4. Evaluar.** Las 100 reseñas humanas se usan al final. Los umbrales no se ajustan después de ver sus etiquetas.')
    st.info('La red de la aplicación usa MLPClassifier de scikit-learn. Difiere de la implementación TensorFlow del notebook: no tiene dropout ni early stopping. Las métricas mostradas pertenecen a esta versión.')
    with st.expander('Ver configuración del experimento'):
        st.json(method)
