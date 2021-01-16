import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--output')
args = parser.parse_args()

import tensorflow as tf
for g in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
import numpy as np
import mydatasets, mymodels, mydatagen

(Xtr, Ytr), tss = getattr(mydatasets, args.dataset)()

model = mymodels.vgg19 if Xtr.shape[1] >= 128 else mymodels.classifier
model, opt = model(Xtr.shape[1:], Ytr.max()+1)
model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy(True)
model.compile(opt, loss, ['accuracy'])

# (1) Train classifier only

g = mydatagen.Gen(Xtr, Ytr, args.batchsize)
model.fit(g.flow(), epochs=args.epochs, verbose=2,
    validation_data=tss[-1], steps_per_epoch=g.steps())

if args.output:
    model.save(args.output)
