import tensorflow as tf
import numpy as np

class Transform:
    def __init__(self):
        self.g = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            shear_range=0.3,
        )

    def each(self, x):
        x = self.g.random_transform(x)
        x = np.minimum(1, x*np.random.uniform(0.8, 1.2))  # brightness
        return x

    def all(self, X):
        return np.array([self.each(x) for x in X])

class Gen:
    def __init__(self, X, Y, batchsize):
        self.t = Transform()
        self.X = X
        self.Y = Y
        self.batchsize = batchsize

    def steps(self):
        return int(np.ceil(int(len(self.X) / self.batchsize)))

    def epoch(self):
        N = len(self.X)
        ix = np.random.choice(N, N, False)
        for i in range(0, N, self.batchsize):
            yield ix[i:i+self.batchsize]

    def flow(self):
        while True:
            for ix in self.epoch():
                yield self.t.all(self.X[ix]), self.Y[ix]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import mydatasets
    X, Y = mydatasets.prates('train')
    g = Gen(X, Y, 12)
    for i, (x, _) in enumerate(g.flow()):
        plt.subplot(2, 6, i+1)
        plt.imshow(x)
    plt.show()
