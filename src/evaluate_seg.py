import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
args = parser.parse_args()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
for g in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
import numpy as np
import mydatasets
from skimage.io import imsave

(Xtr, Ytr), tss = getattr(mydatasets, args.dataset)()
model = tf.keras.models.load_model(args.model, compile=False)
S = model.predict(Xtr)
for i, s in enumerate(S):
    imsave('seg-%d.png' % i, ((s >= 0.5)*255).astype(np.uint8))
