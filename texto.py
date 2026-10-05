import re, html, unicodedata
from functools import lru_cache
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
TEMAS = ["limpieza","ruido","ubicacion","anfitrion","precio"]
SEMILLA=20261004
def limpiar(texto):
    texto = html.unescape(re.sub(r'<[^>]+>', ' ', str(texto))).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', texto)).strip()

PATRONES = {'limpieza': '\\b(limpi\\w*|aseo|higien\\w*|suci\\w*|polvo|moho|humedad|mal olor|cucarach\\w*|clean\\w*|dirty|dirt|dust\\w*|mould|mold|filthy|spotless|propret\\w*|limpez\\w*|sujo\\w*)\\b', 'ruido': '\\b(ruido\\w*|silenci\\w*|bullic\\w*|bulla|escandalo\\w*|acustic\\w*|noise|noisy|loud|loudly|quiet|silent|soundproof\\w*|street noise|barulho\\w*|barulhento\\w*|silencioso\\w*)\\b', 'ubicacion': '\\b(ubic\\w*|hubic\\w*|umbic\\w*|localiz\\w*|location|located|situad\\w*|cerca\\w*|cercan\\w*|centr\\w*|barrio\\w*|sector|zona|metro|subway|neighbou?rhood|bairro|near\\w*|walking distance|walk to|minutes from|transport\\w*|conectividad|entorno|surroundings|vicinity)\\b', 'anfitrion': '\\b(anfitri\\w*|host\\w*|owner\\w*|duen\\w*|propriet\\w*|conserj\\w*|comunica\\w*|communica\\w*|respond\\w*|respuesta\\w*|atent\\w*|amabl\\w*|cordial\\w*|helpful|friendly|responsive|receptiv\\w*|prestativ\\w*|personnel)\\b', 'precio': '\\b(precio\\w*|preco\\w*|cost|costs|costly|costo\\w*|caro|caros|cara|caras|barat\\w*|tarifa\\w*|cobr\\w*|descuent\\w*|reembols\\w*|price\\w*|expensiv\\w*|cheap\\w*|value|money|deal|worth|refund\\w*|fee\\w*)\\b'}
def etiquetas_reglas(texto):
    texto = re.sub(r'\bclean (design|lines)\b','',limpiar(texto))
    return [int(bool(re.search(PATRONES[t],texto))) for t in TEMAS]

PALABRAS_FRECUENTES = set('a al algo algunas algunos ante antes aquel aquella aquellas aquellos aqui asi aun aunque bajo bien cada casi como con contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estaban estado estar estas este esto estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mis mucho muy nada ni nos nuestra nuestras nuestro nuestros o otra otras otro otros para pero por porque que quien quienes se sea ser si sido sin sobre solo son su sus tambien tanto te tiene todo todos tu tus un una unas uno unos usted ustedes ya y the a an and or of in on at to for from with this that it is was were be been as by its our their my your'.split())
stemmer = SnowballStemmer('spanish')
@lru_cache(maxsize=100000)
def raiz(palabra):
    return stemmer.stem(palabra)
def palabras(texto):
    return [raiz(p) for p in limpiar(texto).split() if len(p)>1 and p not in PALABRAS_FRECUENTES and not p.isdigit()]
def longitud(textos):
    return np.log1p(textos.fillna('').astype(str).str.len().to_numpy().reshape(-1,1))
def preparar(ngram=(1,1), red=False, componentes=128, max_features=6000):
    texto = TfidfVectorizer(tokenizer=palabras, token_pattern=None, lowercase=False,
                           min_df=2, max_features=max_features, sublinear_tf=True, ngram_range=ngram)
    if red:
        texto = Pipeline([('tfidf',texto), ('svd',TruncatedSVD(n_components=componentes, random_state=SEMILLA))])
    columnas = ColumnTransformer([('texto',texto,'comments'),
        ('longitud',Pipeline([('medir',FunctionTransformer(longitud)), ('escalar',StandardScaler())]),'comments')],
        sparse_threshold=0 if red else 1.0)
    return Pipeline([('columnas',columnas), ('escala',StandardScaler())]) if red else columnas
