# Clasificación de reseñas · Equipo 6 · Streamlit

Aplicación: https://utpmachinelearning.streamlit.app/

Clasifica reseñas en limpieza, ruido, ubicación, anfitrión y precio. Incluye análisis individual y CSV, comparación de modelos, guía de uso y una sección **Red Keras** con arquitectura, curvas y evidencia del control del sobreajuste.

## Estado de la publicación

El modelo Keras entrenado y su preparador TF-IDF/SVD están incluidos. La interfaz permite seleccionar la red neuronal y consultar su arquitectura, curvas y evidencia en la pestaña Red Keras.

## Nivel 2 — Keras obligatorio

La red es un MLP real en **tf.keras**. TF-IDF se reduce con **TruncatedSVD a 64 componentes**, se agrega la longitud y se escalan las 65 entradas. Arquitectura seleccionada: Dense(64, ReLU) → Dropout(0.20) → Dense(5, sigmoid). Optimizador Adam (0.001), binary_crossentropy, batch 128. EarlyStopping supervisa val_loss, con paciencia 6, min_delta 0.0001 y restore_best_weights=True.

Se ejecutaron 74 de 100 épocas y se restauraron los pesos de la época 68. Una ablación sin dropout ni early stopping mostró menor pérdida de entrenamiento y mayor pérdida de validación. Los controles redujeron 13.1% la pérdida de validación en esta partición. Esto no garantiza un F1 mayor: se documentan ambas métricas sin ocultar diferencias.

Lee [NIVEL_2_KERAS.md](NIVEL_2_KERAS.md) para la arquitectura completa, decisiones, curvas, comparación, límites y pasos de reproducción. La condición de XGBoost/LightGBM es para datos tabulares o series de tiempo y no aplica a este problema de texto.

## Ejecutar

Con Python 3.11 o 3.12, en la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Los tres modelos están disponibles; abrir la aplicación no los vuelve a entrenar. El modelo Keras, su preparador, las curvas y las métricas están incluidos. No necesitas claves API.

## Publicar en Streamlit Community Cloud

En https://share.streamlit.io selecciona el repositorio `nickymarie090490/utpmachinelearnig`, rama `main` y archivo `app.py`. Usa Python 3.11 o 3.12. requirements.txt incluye TensorFlow CPU y Keras. La instalación inicial puede tardar más que la versión anterior.

## Archivos

- `app.py`: interfaz con seis pestañas, incluida guía y Nivel 2.
- `texto.py`: limpieza, reglas y preparación TF-IDF/SVD ajustada en entrenamiento.
- `keras_red.py`: carga segura por integridad e inferencia de la red Keras.
- `entrenar.py`: clásicos, dos candidatos Keras, early stopping, ablación y evaluación.
- `generar_informe.py`: genera informe y figura desde resultados reales.
- `preparar_excel.py`: reconstruye las muestras del Excel original.
- `LogisticRegression.joblib`, `LinearSVC.joblib`: modelos clásicos y sus transformaciones.
- `modelos/red_keras.keras`: red con pesos restaurados por early stopping.
- `modelos/keras_pre.part*`, `keras_manifest.json`: preparador comprimido en partes, con hashes de integridad. Mantén todos los archivos.
- `datos/`: métricas, configuración y curvas. Las reseñas originales no se publican.
- `NIVEL_2_KERAS.md`: documentación de cumplimiento y sobreajuste.

## Reproducir el experimento

Copia tu Excel original `Etiquetados.xlsx` junto a app.py y ejecuta:

```bash
python preparar_excel.py
python entrenar.py
```

`python entrenar.py --solo-red` conserva los clásicos y recalcula la parte neuronal. `python generar_informe.py` regenera la documentación y la figura desde resultados guardados.

Se recorren 690 112 reseñas, se conserva la muestra humana de 100 filas y se excluyen sus alojamientos y copias de sus textos. Se seleccionan 25 000 reseñas con reglas automáticas. Entrenamiento: 19 931; validación: 5 069; grupos sin solapamiento. La red compara dos arquitecturas con igual máximo de épocas y callback. Los clásicos ajustan C y unigramas/bigramas con GroupKFold de tres pliegues. Las etiquetas humanas se consultan solo al final.

## Resultados actuales

| Modelo | Macro F1 humano | Micro F1 humano | Coincidencia completa |
|---|---:|---:|---:|
| Regresión logística | 0.912 | 0.910 | 72% |
| SVM lineal | 0.926 | 0.923 | 76% |
| Red Keras | 0.635 | 0.750 | 47% |

La muestra humana es pequeña y dirigida; no garantiza generalización poblacional. Las etiquetas automáticas son imperfectas. Detectar un tema no determina si el comentario es positivo o negativo. Las puntuaciones no están calibradas.

## Relación con el notebook

Se conserva el planteamiento multietiqueta y la separación por alojamiento. Esta versión vuelve a usar Keras, SVD, dropout y early stopping, pero sus resultados pertenecen a esta ejecución; no son los del notebook original. La app no sustituye los demás análisis académicos del notebook (idiomas, intervalos, importancia por permutación o curvas de tamaño de muestra). La implementación previa de MLPClassifier fue retirada de GitHub.

## Verificación

Se comprobó carga del preparador y la red guardada, integridad SHA-256, cinco salidas, activación real de early stopping, curvas de las épocas ejecutadas, separación de alojamientos e integración Streamlit de los tres modelos. Los modelos se cargan exclusivamente desde el repositorio de confianza; los visitantes no pueden subir archivos de modelos.
