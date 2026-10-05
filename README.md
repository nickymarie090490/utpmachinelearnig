# Streamlit · Problema 2 · Equipo 6

## Publicar sin programar
1. Descomprime el ZIP.
2. Crea una cuenta en https://github.com y un repositorio llamado `resenas-equipo6`.
3. En el repositorio selecciona Add file → Upload files. Sube **el contenido** de la carpeta `streamlit_equipo6`, no el ZIP. Conserva la carpeta `datos`. app.py y requirements.txt deben quedar en la raíz.
4. Abre https://share.streamlit.io e inicia sesión con GitHub.
5. Pulsa Create app, selecciona tu repositorio, rama main y archivo app.py. En Advanced settings selecciona Python 3.11.
6. Pulsa Deploy. Cuando termine, tendrás una URL para compartir.

Los tres archivos .joblib ya contienen modelos entrenados. No hay que abrir Colab ni entrenar en la nube. No necesitas API keys ni Google AI Studio.

Documentación oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Ejecutar en tu Mac / PyCharm
En la terminal, dentro de esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Archivos
- app.py: interfaz con reseña individual, carga CSV, resultados y explicación.
- texto.py: limpieza, stemming español, reglas y TF-IDF, adaptados del notebook.
- entrenar.py: regresión logística, SVM lineal y red neuronal MLP.
- preparar_excel.py: reconstruye las muestras desde Etiquetados.xlsx.
- datos/: métricas agregadas y metodología de esta ejecución. Las muestras de reseñas no se publican en GitHub.
- *.joblib: modelos entrenados con sus transformaciones.

## Reproducir
El repositorio público no incluye muestra_manual.csv ni entrenamiento_debil.csv.gz. Para reconstruirlas utiliza tu Excel original siguiendo los comandos siguientes.

El Excel original no está dentro del ZIP para mantener pequeño el proyecto. Copia Etiquetados.xlsx junto a app.py y ejecuta:

```bash
python preparar_excel.py
python entrenar.py
```

Se toman todas las filas con cinco etiquetas binarias completas como muestra humana; se conserva su etiquetado. Se recorren las 690 112 reseñas; se excluyen alojamientos humanos, copias de sus textos, textos vacíos y duplicados normalizados. Se seleccionan 25 000 reseñas con muestreo de reservorio y se crean etiquetas automáticas con las reglas originales.

Se reserva 20% de alojamientos automáticos para validación. Los clásicos ajustan C y unigramas/bigramas mediante GroupKFold de tres pliegues. Dos arquitecturas MLP (64 y 128/64) se seleccionan con la validación automática fija; no con las etiquetas humanas. Las transformaciones se ajustan dentro del entrenamiento. Las etiquetas humanas se usan solo al final.

## Diferencias con Problema_2.ipynb
Se conserva el planteamiento multietiqueta y la separación por alojamiento. La red usa scikit-learn para simplificar el despliegue; no es el modelo TensorFlow original, no tiene dropout ni early stopping y usa 70 iteraciones. Su selección se hace con una partición agrupada, no con CV de tres pliegues. Las métricas incluidas pertenecen a esta ejecución y no son las del notebook. Esta aplicación no sustituye todos los análisis académicos del notebook (curvas, intervalos, idiomas e importancia por permutación).

Si la red muestra advertencia de convergencia, significa que alcanzó el límite de iteraciones; consulta arquitecturas.csv. No se garantiza convergencia. Las puntuaciones no están calibradas. Detectar temas no identifica sentimiento ni quejas. La muestra manual es pequeña y dirigida. No se necesita que la red sea el mejor modelo: se muestran los resultados reales.

Los modelos joblib solo deben cargarse desde este proyecto de confianza; la aplicación no acepta modelos subidos por visitantes. El ZIP no incluye las columnas de nombres ni identificadores de los autores de reseñas. Los textos originales pueden mencionar personas; revísalos antes de publicar un repositorio público.

## Verificación de esta entrega
- Regresión logística: Macro F1 manual 0.9115.
- SVM lineal: Macro F1 manual 0.9261.
- Red neuronal: Macro F1 manual 0.7115.
- Las dos arquitecturas MLP alcanzaron 70 iteraciones con advertencia de convergencia. La seleccionada fue (128, 64).
- Se comprobó la separación de alojamientos, ausencia de textos humanos en entrenamiento y de textos normalizados duplicados.
- Se ejecutó Streamlit AppTest: carga, cambio de modelo y análisis individual sin excepciones para los tres modelos.
