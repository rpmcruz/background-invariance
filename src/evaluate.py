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

(Xtr, Ytr), tss = getattr(mydatasets, args.dataset)()
model = tf.keras.models.load_model(args.model, compile=False)

acc_tr = np.mean(model.predict(Xtr).argmax(1) == Ytr)
acc_tss = [np.mean(model.predict(Xts).argmax(1) == Yts) for Xts, Yts in tss]
acc_ts = ','.join(map(str, acc_tss))
print(f'{acc_tr},{acc_ts}')
