"""Pruebas de contratos y prevención de fuga; no sustituyen evaluación humana."""
import io,json,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd,pytest
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.metrics import multilabel_confusion_matrix,f1_score
from texto import limpiar,palabras,etiquetas_reglas,preparar,TEMAS,SEMILLA
from inferencia import validar_textos,analizar_modelo,explicar_local
from keras_red import cargar_keras
P=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('texto,esperado',[('<b>¡LÍMPIO!</b>','limpio'),('  RUIDO\n calle  ','ruido calle'),(None,'none')])
def test_normalizacion(texto,esperado):assert limpiar(texto)==esperado

def test_stopwords_stemming():assert palabras('Hay mucho ruido y casas')==['ruid','cas']

@pytest.mark.parametrize('texto,pos',[('hay mucho ruido',[0,1,0,0,0]),('Costanera Center',[0,0,0,0,0]),('el precio es caro',[0,0,0,0,1]),('loud street noise',[0,1,0,0,0]),('clean and noisy',[1,1,0,0,0])])
def test_reglas(texto,pos):assert etiquetas_reglas(texto)==pos

@pytest.mark.parametrize('textos',[[],[''],['123 !'],[None],['a'*15001],['ok']*5001])
def test_entrada_invalida(textos):
 with pytest.raises(ValueError):validar_textos(textos)

def test_vectorizar_solo_entrenamiento():
 x=pd.DataFrame({'comments':['limpio ruido','limpio metro','ruido metro','limpio ruido metro']})
 pre=preparar(max_features=30);pre.fit(x);pre.transform(pd.DataFrame({'comments':['xenotermino'] }))
 assert 'xenotermin' not in pre.named_transformers_['texto'].vocabulary_

def test_svd_denso_y_estructurada():
 x=pd.DataFrame({'comments':['limpio ruido','limpio metro','ruido metro','limpio ruido metro']})
 pre=preparar(red=True,componentes=2,max_features=30);z=pre.fit_transform(x)
 assert z.shape==(4,3) and isinstance(z,np.ndarray) and np.isfinite(z).all()

def test_entidades_separadas():
 d=pd.read_csv(P/'datos/entrenamiento_debil.csv.gz',dtype={'listing_id':str});m=pd.read_csv(P/'datos/muestra_manual.csv',dtype={'listing_id':str})
 a,b=next(GroupShuffleSplit(test_size=.2,random_state=SEMILLA).split(d,groups=d.listing_id))
 assert not set(d.listing_id)&set(m.listing_id)
 assert not set(d.iloc[a].listing_id)&set(d.iloc[b].listing_id)
 assert not set(d.comments.map(limpiar))&set(m.comments.map(limpiar))
 for a,b in GroupKFold(3).split(d,groups=d.listing_id):assert not set(d.iloc[a].listing_id)&set(d.iloc[b].listing_id)

def test_metrica_y_confusion():
 y=np.array([[1,0],[0,1],[0,0]]);p=np.array([[1,1],[0,0],[0,0]])
 assert f1_score(y,p,average='macro')==.5
 assert multilabel_confusion_matrix(y,p).tolist()==[[[2,0],[0,1]],[[1,1],[1,0]]]

@pytest.mark.parametrize('nombre',['LogisticRegression','LinearSVC','MLP'])
def test_inferencia_y_explicacion(nombre):
 m=cargar_keras(P) if nombre=='MLP' else joblib.load(P/(nombre+'.joblib'))
 p,s=analizar_modelo(m,['hay mucho ruido','limpio y cerca del metro'])
 assert p.shape==(2,5) and np.isin(p,[0,1]).all()
 if s is not None:assert s.shape==(2,5) and np.isfinite(s).all()
 e=explicar_local(m,'hay mucho ruido',nombre);assert not e.empty and np.isfinite(e.Efecto).all()
 if nombre in ('LogisticRegression','MLP'):assert p[0,1]==1 # Regresión sobre fallo observado en producción.

def test_integridad_modelo(tmp_path):
 shutil.copytree(P/'modelos',tmp_path/'modelos');manifest=json.loads((tmp_path/'modelos/keras_manifest.json').read_text())
 part=tmp_path/'modelos'/manifest['preparacion_parts'][0]['archivo'];part.write_bytes(b'corrupto')
 with pytest.raises(ValueError,match='incompleta'):cargar_keras(tmp_path)

def test_arquitectura_early_stopping():
 e=json.loads((P/'datos/evidencia_sobreajuste.json').read_text());h=pd.read_csv(P/'datos/perdida_keras.csv');m=cargar_keras(P)
 assert len(h)==e['epocas_ejecutadas']<=e['max_epocas'];assert 1<=e['mejor_epoca']<=len(h)
 assert e['early_stopping_activo'] and e['grupos_compartidos']==0
 assert m.red.output_shape[-1]==5 and m.red.layers[-1].activation.__name__=='sigmoid'
 assert len([v for v in m.red.layers if v.__class__.__name__=='Dense'])==e['capas_densas']

def test_curva_cuenta_grupos_y_particion():
 df=pd.read_csv(P/'datos/curva_aprendizaje.csv');cv=pd.read_csv(P/'datos/particiones.csv')
 for _,g in df.groupby('modelo'):
  assert g.proporcion.tolist()==[.1,.25,.5,.75,1.];assert g.grupos.is_monotonic_increasing
 assert cv[cv.validador=='GroupKFold correcto'].grupos_compartidos.eq(0).all()
 assert cv[cv.validador=='KFold ingenuo'].grupos_compartidos.gt(0).all()
