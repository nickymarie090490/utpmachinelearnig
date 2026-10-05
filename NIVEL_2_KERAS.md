# Nivel 2 Keras y evaluación final

TF-IDF (max_features=6000, min_df=2, unigramas) → TruncatedSVD(256) + log-longitud → StandardScaler. Entrada: 257. Capas ocultas: [128, 64], ReLU; dropout 0.2; salida 5 sigmoid; Adam 0.001; batch128; pérdida binary crossentropy ponderada. Pesos positivos: [1.854180974252965, 4.890168708746152, 0.9206992074967176, 1.1453876073802436, 4.58245526407412].

EarlyStopping: val_loss, paciencia6, min_delta0.0001, restore_best_weights=True. Épocas: 12/100. Pesos restaurados: época 6. Parada anticipada realmente activada: True.

## Sobreajuste
La BCE no ponderada de validación es 0.109414 con controles y 0.302766 sin dropout/early stopping. Cambio relativo: 63.9%. Se conserva arquitectura, ponderación, semilla, datos y optimizador. Las curvas muestran la pérdida de entrenamiento utilizada; las cifras anteriores usan inferencia sin dropout. No se aísla causalmente cada control ni se garantiza mejor F1. Ablación Macro F1 automático: 0.783.

## Validez
Entrenamiento 19931 filas/6440 alojamientos; validación 5069 filas/1610 alojamientos; test100 reseñas/90 alojamientos. Cero grupos compartidos. Preprocesamiento dentro del pipeline. Las etiquetas automáticas no son verdad humana. Las100 manuales son dirigidas y ya se consultaron en versiones anteriores: esta evaluación no es completamente ciega. No se ajustó el modelo con ellas en esta ejecución. La fecha del dump no consta en el archivo proporcionado.

## Evidencia guardada
- metricas.csv: Macro/MicroF1, coincidencia, entrenamiento, inferencia, tamaño e intervalos por alojamiento.
- particiones.csv: ingenua versus GroupKFold, mismo candidato y tres pliegues.
- curva_aprendizaje.csv:10%,25%,50%,75%,100% de entidades.
- confusion.csv: matrices2x2 por tema y modelo.
- sesgos_idioma.csv: métricas, muestra y grupos por idioma estimado.
- terminos_clasicos.csv e importancia_red.csv: coeficientes y permutation importance.

## Reproducir
python preparar_excel.py --fuente Etiquetados.xlsx
python entrenar.py
python -m pytest -q
streamlit run app.py

Referencias: https://keras.io/api/callbacks/early_stopping/ ; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html ; https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
