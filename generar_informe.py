"""Figuras y resumen metodológico generados con resultados medidos."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
COLORS={'LogisticRegression':'#2563eb','LinearSVC':'#0f766e','MLP':'#d97706'}
def generar():
 e=json.loads((P/'datos/evidencia_sobreajuste.json').read_text());h=pd.read_csv(P/'datos/perdida_keras.csv');b=pd.read_csv(P/'datos/perdida_sin_control.csv')
 fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
 for ax,df,title in zip(axs,[h,b],['Dropout y early stopping','Sin ambos controles']):
  ax.plot(df.epoca,df.Entrenamiento,label='Entrenamiento',color='#2563eb');ax.plot(df.epoca,df.Validación,label='Validación',color='#d97706');ax.set(xlabel='Época',title=title);ax.grid(alpha=.18);ax.legend(frameon=False)
 axs[0].axvline(e['mejor_epoca'],color='#0f766e',linestyle='--');axs[0].set_ylabel('Pérdida usada para entrenar');fig.tight_layout();fig.savefig(P/'datos/curvas_perdida.png',dpi=180);plt.close(fig)
 df=pd.read_csv(P/'datos/curva_aprendizaje.csv');fig,ax=plt.subplots(figsize=(7,3.5))
 for n,gr in df.groupby('modelo'):ax.plot(gr.grupos,gr['Macro F1 automático'],marker='o',label=n,color=COLORS[n])
 ax.set(xlabel='Alojamientos de entrenamiento',ylabel='Macro F1 en validación automática');ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'datos/curva_aprendizaje.png',dpi=180);plt.close(fig)
 cm=pd.read_csv(P/'datos/confusion.csv');fig,axs=plt.subplots(3,5,figsize=(11,6))
 for row,n in enumerate(COLORS):
  for col,(_,v) in enumerate(cm[cm.modelo==n].iterrows()):
   a=np.array([[v.TN,v.FP],[v.FN,v.TP]]);ax=axs[row,col];ax.imshow(a,cmap='Blues');ax.set_xticks([0,1],['0','1']);ax.set_yticks([0,1],['0','1']);ax.set_title(v.tema,fontsize=9)
   if col==0:ax.set_ylabel(n+'\nReal',fontsize=8)
   if row==2:ax.set_xlabel('Predicho',fontsize=8)
   for (i,j),val in np.ndenumerate(a):ax.text(j,i,str(val),ha='center',va='center',color='white' if val>a.max()*.6 else '#111827')
 fig.tight_layout();fig.savefig(P/'datos/matrices_confusion.png',dpi=180);plt.close(fig)
 terms=pd.read_csv(P/'datos/terminos_clasicos.csv');fig,axs=plt.subplots(1,5,figsize=(12,3.4))
 for ax,(tema,gr) in zip(axs,terms[terms.modelo=='LogisticRegression'].groupby('tema',sort=False)):
  gr=gr.head(6).iloc[::-1];ax.barh(gr.termino,gr.coeficiente,color='#2563eb');ax.set_title(tema);ax.tick_params(axis='y',labelsize=8)
 fig.tight_layout();fig.savefig(P/'datos/terminos_clasicos.png',dpi=180);plt.close(fig)
 im=pd.read_csv(P/'datos/importancia_red.csv').nlargest(10,'caida_Macro_F1').iloc[::-1];fig,ax=plt.subplots(figsize=(7,3.5));ax.barh(im.variable,im.caida_Macro_F1,xerr=im.desviacion,color='#d97706');ax.set_xlabel('Caída de Macro F1 al permutar la entrada');fig.tight_layout();fig.savefig(P/'datos/importancia_red.png',dpi=180);plt.close(fig)
 red=(e['loss_val_sin_control']-e['loss_val_restaurada'])/e['loss_val_sin_control']*100
 texto=f'''# Nivel 2 Keras y evaluación final

TF-IDF (max_features=6000, min_df=2, unigramas) → TruncatedSVD({e['svd']}) + log-longitud → StandardScaler. Entrada: {e['entrada_dimensiones']}. Capas ocultas: {e['capas']}, ReLU; dropout {e['dropout']}; salida 5 sigmoid; Adam 0.001; batch128; pérdida {e['loss']}. Pesos positivos: {e['pesos_positivos']}.

EarlyStopping: val_loss, paciencia6, min_delta0.0001, restore_best_weights=True. Épocas: {e['epocas_ejecutadas']}/100. Pesos restaurados: época {e['mejor_epoca']}. Parada anticipada realmente activada: {e['parada_anticipada']}.

## Sobreajuste
La BCE no ponderada de validación es {e['loss_val_restaurada']:.6f} con controles y {e['loss_val_sin_control']:.6f} sin dropout/early stopping. Cambio relativo: {red:.1f}%. Se conserva arquitectura, ponderación, semilla, datos y optimizador. Las curvas muestran la pérdida de entrenamiento utilizada; las cifras anteriores usan inferencia sin dropout. No se aísla causalmente cada control ni se garantiza mejor F1. Ablación Macro F1 automático: {e['F1_val_sin_control']:.3f}.

## Validez
Entrenamiento {e['train']} filas/{e['train_grupos']} alojamientos; validación {e['validacion']} filas/{e['validacion_grupos']} alojamientos; test100 reseñas/{e['test_grupos']} alojamientos. Cero grupos compartidos. Preprocesamiento dentro del pipeline. Las etiquetas automáticas no son verdad humana. Las100 manuales son dirigidas y ya se consultaron en versiones anteriores: esta evaluación no es completamente ciega. No se ajustó el modelo con ellas en esta ejecución. La fecha del dump no consta en el archivo proporcionado.

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
'''
 (P/'NIVEL_2_KERAS.md').write_text(texto)
if __name__=='__main__':generar()
