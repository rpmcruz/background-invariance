import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('binarize_mask', choices=['otsu', '0.5', 'quantile', 'none'])
parser.add_argument('--loss', choices=['abs', 'squared'], default='abs')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--area', type=float, default=0.2)
parser.add_argument('--total-variation', type=float, default=0.1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--output')
args = parser.parse_args()

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
loss = tf.keras.losses.SparseCategoricalCrossentropy(True)

unet, opt = mymodels.unet(Xtr.shape[1:])
unet.summary()

def binarize_mask(mask):
    if args.binarize_mask == 'otsu':
        return mask >= threshold_otsu(mask)
    if args.binarize_mask == '0.5':
        return mask >= 0.5
    if args.binarize_mask == 'quantile':
        return mask >= np.quantile(mask, 1-args.area)
    return mask

# (2) Train masks only

def total_variation(X):
    dx = tf.reduce_mean(tf.abs(X[:, 1:, :] - X[:, :-1, :]))
    dy = tf.reduce_mean(tf.abs(X[:, :, 1:] - X[:, :, :-1]))
    return dx + dy

@tf.function
def train_step(X, Y):
    with tf.GradientTape() as tape:
        M = unet(X, training=True)
        Z = X*M
        Yhat = model(Z, training=True)
        l = loss(Y, Yhat)
        mask_loss = l
        if args.loss == 'abs':
            mask_loss += tf.reduce_mean(tf.nn.relu(tf.reduce_mean(M, [1, 2, 3]) - args.area))
        if args.loss == 'squared':
            mask_loss += tf.reduce_mean(tf.nn.relu(tf.reduce_mean(M, [1, 2, 3]) - args.area)**2)
        mask_loss += args.total_variation*total_variation(M)
    # update masks
    g = tape.gradient(mask_loss, unet.trainable_variables)
    opt.apply_gradients(zip(g, unet.trainable_variables))
    return (l, mask_loss), g

g = mydatagen.Gen(Xtr, Ytr, args.batchsize)

for epoch in range(args.epochs):
    print(f'* Epoch {epoch+1}/{args.epochs}', file=sys.stderr)
    tic = time()
    avg_loss = np.zeros(2)
    for ix in g.epoch():
        ls, grads = train_step(g.t.all(Xtr[ix]), Ytr[ix])
        avg_loss += np.array(ls) / g.steps()
    print('Elapsed time: %ds' % (time()-tic), file=sys.stderr)
    print('Loss - ce: %f, mask: %f' % tuple(avg_loss), file=sys.stderr)
    acc_tr = np.mean(model.predict(Xtr, args.batchsize).argmax(1) == Ytr)
    acc_tss = [np.mean(model.predict(Xts, args.batchsize).argmax(1) == Yts) for Xts, Yts in tss]
    acc_ts = ','.join(map(str, acc_tss))
    print(f'Accuracy - train: {acc_tr}, test: {acc_ts}', file=sys.stderr)
    gc.collect()  # apparently necessary to keep memory from blowing
    if args.debug:
        plt.clf()
        ix = np.linspace(0, len(Xtr), 6, False, dtype=int)
        debug_X = Xtr[ix]
        debug_M = unet(debug_X)
        for i, (x, m) in enumerate(zip(debug_X, debug_M)):
            plt.subplot(3, 6, i+1)
            plt.imshow(x)
            plt.axis('off')

            plt.subplot(3, 6, i+7)
            plt.imshow(m[..., 0], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

            plt.subplot(3, 6, i+13)
            prev_m = m.numpy()
            m = binarize_mask(prev_m)
            plt.imshow(m[..., 0], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

            imsave(f'{prefix}-debug-image-{i}.png', (x*255).astype(np.uint8))
            imsave(f'{prefix}-debug-mask-{i}.png', (prev_m[..., 0]*255).astype(np.uint8))
            imsave(f'{prefix}-debug-post-{i}.png', (m[..., 0]*255).astype(np.uint8))

        plt.suptitle(f'Epoch {epoch+1}')
        plt.savefig(f'{prefix}-mask-epoch-{epoch+1}.png')

print(f'{prefix},{acc_tr},{acc_ts}')
if args.output:
    unet.save(args.output)
