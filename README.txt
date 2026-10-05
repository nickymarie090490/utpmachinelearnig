# Problema 2 · Clasificación multietiqueta de reseñas

Aplicación pública: https://utpmachinelearning.streamlit.app/
Repositorio: https://github.com/nickymarie090490/utpmachinelearnig

Detecta menciones de limpieza, ruido, ubicación, anfitrión y precio. Un elogio, una negación o una crítica pueden mencionar un tema. No clasifica sentimiento ni mide quejas. Modelo de producción elegido: **regresión logística** por su salida probabilística, explicación lineal y costo. SVM y Keras se conservan para comparación académica.

## Ejecutar la app

Python 3.11 o 3.12, preferiblemente 3.12.14 (ejecución del informe):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Los modelos entrenados están incluidos. No requiere API ni entrenamiento al abrir la app. Las salidas sigmoid/logísticas son puntuaciones no calibradas, con umbral 0.5. SVM muestra decisiones y márgenes, sin porcentajes de probabilidad.

## Reproducir de principio a fin

El Excel original y las reseñas no se publican. Coloca tu archivo `Etiquetados.xlsx` en la carpeta del proyecto y ejecuta:

```bash
python preparar_excel.py --fuente Etiquetados.xlsx
python entrenar.py
python -m pytest -q --junitxml=datos/pruebas_unitarias.xml
streamlit run app.py
```

`preparar_excel.py` reconstruye 100 reseñas etiquetadas y una muestra automática de 25.000. `entrenar.py` realiza GridSearchCV agrupado, comparación de particiones, tres candidatos Keras, early stopping, ablación, curva de aprendizaje, evaluación humana final, costos, matrices, errores, interpretabilidad e idiomas. `generar_informe.py` reconstruye figuras y resumen a partir de resultados. Fija semilla 20261004 y limita hilos para repetibilidad; tiempos y pequeños redondeos pueden variar según hardware.

## Resultados finales de esta versión

| Modelo | Macro F1 manual | Micro F1 | Coincidencia completa | Búsqueda y entrenamiento s | Inferencia ms por reseña | MB en disco |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | 0.913 | 0.912 | 74% | 14.206 | 0.064 | 0.297 |
| SVM lineal | 0.934 | 0.932 | 80% | 14.942 | 0.067 | 0.297 |
| Keras MLP | 0.781 | 0.821 | 51% | 25.809 | 0.101 | 12.128 |

Baseline mayoritario por etiqueta: Macro F1 0.140. Inferencia en lotes de100 tras calentamiento, mediana de7 repeticiones. Costos de entrenamiento incluyen búsqueda y refit, excluyen análisis auxiliares. Todos usan los mismos textos, longitud y partición.

La red seleccionada: TF-IDF (unigramas, min_df2, máximo6000 términos) → TruncatedSVD256 + log-longitud → StandardScaler → Dense128 ReLU → Dropout0.20 → Dense64 ReLU → Dropout0.20 → Dense5 sigmoid. Adam0.001, batch128, BCE ponderada por etiqueta con pesos calculados solo en entrenamiento. EarlyStopping ejecutó12/100 épocas y restauró6. La ablación sin dropout y parada obtuvo BCE de validación0.303 frente a0.109 con controles. No demuestra que el F1 mejore por cada control por separado.

La prueba «hay mucho ruido» ahora detecta ruido en los tres modelos. Esto no garantiza acierto general: Keras detectó solo9 de25 menciones de ruido del test.

## Alcance y límites

- Santiago según el notebook suministrado. El Excel contiene reseñas entre2010-11-13 y2026-07-01. **La fecha del dump no fue registrada** y no debe confundirse con fecha de reseña o modificación del Excel. Debe completarse con el origen de descarga del equipo.
- 100 reseñas manuales dirigidas de90 alojamientos; ya fueron consultadas en versiones anteriores. No son un test completamente ciego ni representativo. No se ajustaron parámetros con ellas en esta ejecución. Hace falta un nuevo test humano independiente para cumplir estrictamente esa condición de la guía.
- Entrenamiento19931 filas/6440 alojamientos, validación5069/1610; ningún alojamiento o texto humano normalizado se usa para entrenar. Todo el preprocesamiento se ajusta dentro de pipelines después de separar datos.
- Las etiquetas débiles enseñan reglas imperfectas. La comparación ingenua y agrupada muestra diferencias pequeñas de F1, pese a2690 alojamientos compartidos en el KFold ingenuo.
- Curvas al10/25/50/75/100% de entidades. Los clásicos siguen mejorando al100%; los datos no permiten afirmar suficiencia. Keras es inestable.
- Detección automática de idioma; inglés16 reseñas y español67, portugués1 sin base para inferir sesgo robusto. No se usan identidades personales.

## Despliegue reproducible

En Streamlit Community Cloud, conecta este repositorio, selecciona rama main y archivo app.py, Python3.12, e instala requirements.txt. Guarda los modelos junto a su preparador y manifiesto; las sumas SHA-256 detectan daños. Actualizar main vuelve a desplegar la app. Verifica análisis individual, CSV, comparación, guía y sección Red Keras.

## Evidencia y entrega

`datos/` contiene resultados agregados y figuras, no reseñas publicadas. `tests/test_proyecto.py` verifica25 casos de contratos, limpieza, entradas, separación por alojamiento/texto, pipelines, matriz de confusión, salida, explicación local, integridad y regresión del fallo de ruido. Las pruebas unitarias no demuestran calidad semántica general.

El informe Word/PDF y la bitácora se entregan por separado. La exposición oral de12 minutos sigue siendo responsabilidad del equipo. No se añadió boosting porque el problema es texto y esa obligación corresponde a tabulares o series de tiempo.
