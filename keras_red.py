"""Inferencia del MLP Keras y su preparación TF-IDF → SVD."""
from pathlib import Path
import hashlib
import io
import json
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import joblib
import numpy as np

class RedKeras:
    def __init__(self, preparacion, red):
        self.preparacion = preparacion
        self.red = red

    def predict_proba(self, x):
        z = self.preparacion.transform(x).astype('float32')
        return np.vstack([np.asarray(self.red(z[i:i+256], training=False)) for i in range(0, len(z), 256)])

    def predict(self, x):
        return (self.predict_proba(x) >= .5).astype(int)


def cargar_keras(carpeta):
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass
    p = Path(carpeta)/'modelos'
    manifest = json.loads((p/'keras_manifest.json').read_text())
    blocks = []
    for item in manifest['preparacion_parts']:
        raw = (p/item['archivo']).read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('La preparación del modelo Keras está incompleta.')
        blocks.append(raw)
    raw = b''.join(blocks)
    if hashlib.sha256(raw).hexdigest() != manifest['preparacion_sha256']:
        raise ValueError('La preparación Keras no supera la verificación de integridad.')
    model_bytes = (p/'red_keras.keras').read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != manifest['red_sha256']:
        raise ValueError('El modelo Keras no supera la verificación de integridad.')
    preparacion = joblib.load(io.BytesIO(raw))
    red = tf.keras.models.load_model(p/'red_keras.keras', compile=False)
    return RedKeras(preparacion, red)
