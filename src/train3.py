import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('unet_model')
parser.add_argument('binarize_mask', choices=['otsu', '0.5', 'quantile', 'none'])
parser.add_argument('back', choices=['black', 'noise', 'adv', 'real'])
parser.add_argument('--tiles', type=int, default=2)
parser.add_argument('--area', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--imsave', action='store_true')
parser.add_argument('--output')
args = parser.parse_args()
assert args.tiles >= 1

import tensorflow as tf
for g in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 4)
from skimage.io import imsave
from time import time
import mydatasets, mymodels, mydatagen
import sys, gc
prefix = '-'.join(sys.argv)

(Xtr, Ytr), tss = getattr(mydatasets, args.dataset)()

model = tf.keras.models.load_model(args.model, compile=False)
opt = tf.keras.optimizers.Adam(1e-5 if Xtr.shape[1] >= 128 else 1e-3)
loss = tf.keras.losses.SparseCategoricalCrossentropy(True)

if args.unet_model == 'heuristic':
    def unet(X):
        return tf.constant(X > 0)
else:
    unet = tf.keras.models.load_model(args.unet_model, compile=False)

if args.back == 'adv':
    adv, adv_opt, adv_noise = mymodels.gen(Xtr.shape[1:], args.tiles)
    adv.summary()
elif args.back == 'real':
    back_gen = mydatasets.back_gen()

# (3) Train classifier+background

def adv_back(n):
    full_B = []
    for _ in range(args.tiles):
        line_B = []
        for _ in range(args.tiles):
            noise = tf.random.normal([n, adv_noise])
            each_B = adv(noise, training=True)
            line_B.append(each_B)
        line_B = tf.concat(line_B, 2)
        full_B.append(line_B)
    return tf.concat(full_B, 1)

def merge_back(X, M, B):
    if args.back == 'black':
        return X*M
    elif args.back == 'noise':
        N = tf.random.uniform(X.shape)
        return X*M + N*(1-M)
    elif args.back == 'adv':
        return X*M + adv_back(X.shape[0])*(1-M)
    elif args.back == 'real':
        return X*M + B*(1-M)

@tf.function
def train_step(X, Y, M, B):
    with tf.GradientTape(args.back == 'adv') as tape:
        if args.back == 'black':
            B = 0
        elif args.back == 'noise':
            B = tf.random.normal(X.shape)
        elif args.back == 'adv':
            B = adv_back(X.shape[0])
        Z = X*M + B*(1-M)

        Yhat = model(Z, training=True)
        l = loss(Y, Yhat)
        adv_loss = -l
        if Xtr.shape[3] == 1 and args.back == 'adv':
            # for mnist and fashion_mnist, avoid backgrounds that are all white
            adv_loss += tf.nn.relu(tf.reduce_mean(B) - 0.5)
    # update classifier
    g = tape.gradient(l, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    # update background
    if args.back == 'adv':
        g = tape.gradient(adv_loss, adv.trainable_variables)
        adv_opt.apply_gradients(zip(g, adv.trainable_variables))
    else:
        adv_loss = 0
    del tape
    return l, adv_loss

def binarize_mask(mask):
    if args.binarize_mask == 'otsu':
        return mask >= threshold_otsu(mask)
    if args.binarize_mask == '0.5':
        return mask >= 0.5
    if args.binarize_mask == 'quantile':
        return mask >= np.quantile(mask, 1-args.area)
    return mask

g = mydatagen.Gen(Xtr, Ytr, args.batchsize)

for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1}/{args.epochs}', file=sys.stderr)
    tic = time()
    avg_loss = np.zeros(2)
    for ix in g.epoch():
        M = unet(Xtr[ix]).numpy()
        M = np.array([binarize_mask(m) for m in M], np.float32)
        if args.back == 'real':
            B = np.array([next(back_gen) for _ in range(len(ix))])
        else:
            B = None
        ls = train_step(g.t.all(Xtr[ix]), Ytr[ix], M, B)
        avg_loss += np.array(ls) / g.steps()
    print('Elapsed time: %ds' % (time()-tic), file=sys.stderr)
    print('Loss - ce: %f, adv: %f' % tuple(avg_loss), file=sys.stderr)
    acc_tr = np.mean(model.predict(Xtr, args.batchsize).argmax(1) == Ytr)
    acc_tss = [np.mean(model.predict(Xts, args.batchsize).argmax(1) == Yts) for Xts, Yts in tss]
    acc_ts = ','.join(map(str, acc_tss))
    print(f'Accuracy - train: {acc_tr}, test: {acc_ts}', file=sys.stderr)
    gc.collect()  # apparently necessary to keep memory from blowing
    if args.imsave:
        for X, Y in tss:
            X = X[:12]
            M1 = unet(X).numpy()
            M2 = np.array([binarize_mask(m) for m in M1], np.float32)
            Z = merge_back(X, M2, adv_back(len(X)))
            for i, (x, m1, m2, z) in enumerate(zip(X, M1, M2, Z)):
                imsave(f'x-dataset-{args.dataset}-tiles-{args.tiles}-epoch-{epoch+1}-i-{i}.png', (255-x*255).astype(np.uint8))
                imsave(f'm1-dataset-{args.dataset}-tiles-{args.tiles}-epoch-{epoch+1}-i-{i}.png', (255-m1*255).astype(np.uint8))
                imsave(f'm2-dataset-{args.dataset}-tiles-{args.tiles}-epoch-{epoch+1}-i-{i}-{args.binarize_mask}.png', (255-m2*255).astype(np.uint8))
                imsave(f'z-dataset-{args.dataset}-tiles-{args.tiles}-epoch-{epoch+1}-i-{i}.png', (255-z*255).numpy().astype(np.uint8))

if args.output:
    model.save(args.output)
    if args.back == 'adv':
        adv.save(f'{args.output[:-3]}-back.h5')
