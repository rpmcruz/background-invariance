import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import disk, binary_dilation
import mybackgrounds

def load_prates(subfolder, is_mask):
    datadir = os.path.expanduser('~/.keras/datasets/prates')
    if not os.path.exists(datadir):
        tf.keras.utils.get_file('prates.zip',
            'https://www.dropbox.com/sh/u5fmpv68dba1nxz/AAAwbHfmXQPUOtYQARut0fnJa?dl=1',
            cache_subdir=datadir, extract=True)
    imgs_dir = os.path.join(datadir, 'components', subfolder)
    imgs = sorted(os.listdir(imgs_dir))
    if is_mask:
        M = [imread(os.path.join(imgs_dir, mask), True) for mask in masks]
        M = [m[:, :, np.newaxis] / m.max() for m in M]
    else:
        X = [imread(os.path.join(imgs_dir, img)) / 255 for img in imgs]
    return X

def prates():
    Xtr = []
    Ytr = []
    Xts1 = []
    Yts1 = []
    Xts2 = []
    Yts2 = []
    for types, X, Y in (([1], Xtr, Ytr), ([3], Xts1, Yts1), ([2, 4], Xts2, Yts2)):
        for type in types:
            for y, material in enumerate(['IPL', 'IPN', 'IPV', 'ISB']):
                x = load_prates(f'data_{material}_{type}', False)
                X += x
                Y += [y]*len(x)
    Xtr = np.array(Xtr, np.float32)
    Ytr = np.array(Ytr, np.int32)
    Xts1 = np.array(Xts1, np.float32)
    Yts1 = np.array(Yts1, np.int32)
    Xts2 = np.array(Xts2, np.float32)
    Yts2 = np.array(Yts2, np.int32)
    return (Xtr, Ytr), [(Xts1, Yts1), (Xts2, Yts2)]

def prates_masks():
    M = []
    for type in [1, 3]:
        for material in enumerate(['IPL', 'IPN', 'IPV', 'ISB']):
            M += load_prates(f'data_{material}_{type}', False)
    return np.array(M, np.float32)

def prates_backgen():
    bg_dir = os.path.join(os.path.expanduser('~/.keras/datasets/prates'), 'bg')
    B = [imread(os.path.join(bg_dir, f)).astype(np.float32)/255 for f in os.listdir(bg_dir)]
    while True:
        b = B[np.random.choice(len(B))]
        x = np.random.randint(b.shape[1]-224)
        y = np.random.randint(b.shape[0]-224)
        yield b[y:y+224, x:x+224]

def mnist():
    (Xtr, Ytr), (Xts, Yts) = tf.keras.datasets.mnist.load_data()
    Xtr = np.pad(Xtr, ((0, 0), (2, 2), (2, 2))).astype(np.float32)[..., np.newaxis] / 255
    Xts = np.pad(Xts, ((0, 0), (2, 2), (2, 2))).astype(np.float32)[..., np.newaxis] / 255
    Ytr = Ytr.astype(np.int32)
    Yts = Yts.astype(np.int32)
    return (Xtr, Ytr), [(Xts, Yts)]

def fashionmnist():
    (Xtr, Ytr), (Xts, Yts) = tf.keras.datasets.fashion_mnist.load_data()
    Xtr = np.pad(Xtr, ((0, 0), (2, 2), (2, 2))).astype(np.float32)[..., np.newaxis] / 255
    Xts = np.pad(Xts, ((0, 0), (2, 2), (2, 2))).astype(np.float32)[..., np.newaxis] / 255
    Ytr = Ytr.astype(np.int32)
    Yts = Yts.astype(np.int32)
    return (Xtr, Ytr), [(Xts, Yts)]

def cifar10():
    (Xtr, Ytr), (Xts, Yts) = tf.keras.datasets.cifar10.load_data()
    Xtr = np.array([resize(x, (64, 64)) for x in Xtr], np.float32)
    Xts = np.array([resize(x, (64, 64)) for x in Xts], np.float32)
    Ytr = Ytr.astype(np.int32)[:, 0]
    Yts = Yts.astype(np.int32)[:, 0]
    return (Xtr, Ytr), [(Xts, Yts)]

def stl10():
    # http://ai.stanford.edu/~acoates/stl10/
    datadir = os.path.expanduser('~/.keras/datasets/stl10_matlab')
    if not os.path.exists(datadir):
        tf.keras.utils.get_file('stl10_matlab.tar.gz',
            'http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz',
            extract=True)

    d = loadmat(os.path.join(datadir, 'train.mat'))
    Xtr = d['X'].reshape((-1, 96, 96, 3), order='F')
    Xtr = np.array([resize(x, (64, 64)).astype(np.float32) for x in Xtr])
    Ytr = d['y'][:, 0].astype(np.int32) - 1

    d = loadmat(os.path.join(datadir, 'test.mat'))
    Xts = d['X'].reshape((-1, 96, 96, 3), order='F')
    Xts = np.array([resize(x, (64, 64)).astype(np.float32) for x in Xts])
    Yts = d['y'][:, 0].astype(np.int32) - 1
    return (Xtr, Ytr), [(Xts, Yts)]

def svhn():
    # http://ufldl.stanford.edu/housenumbers/
    datadir = os.path.expanduser('~/.keras/datasets/svhn')
    if not os.path.exists(datadir):
        tf.keras.utils.get_file('train_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            cache_subdir=datadir, extract=True)
        tf.keras.utils.get_file('test_32x32.mat',
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            cache_subdir=datadir, extract=True)

    d = loadmat(os.path.join(datadir, 'train_32x32.mat'))
    Xtr = np.rollaxis(d['X'], -1).astype(np.float32) / 255
    Ytr = d['y'][:, 0].astype(np.int32) % 10

    d = loadmat(os.path.join(datadir, 'test_32x32.mat'))
    Xts = np.rollaxis(d['X'], -1).astype(np.float32) / 255
    Yts = d['y'][:, 0].astype(np.int32) % 10
    return (Xtr, Ytr), [(Xts, Yts)]

def synthdigits():
    # http://yaroslav.ganin.net/
    datadir = os.path.expanduser('~/.keras/datasets/synthdigits')
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
        url = 'https://drive.google.com/uc?id=0B9Z4d7lAwbnTSVR1dEFSRUFxOUU'
        os.system(f'gdown "{url}" -O {datadir}/SynthDigits.zip')
        os.system(f'unzip {datadir}/SynthDigits.zip -d {datadir}')

    d = loadmat(os.path.join(datadir, 'synth_train_32x32.mat'))
    Xtr = np.rollaxis(d['X'], -1).astype(np.float32) / 255
    Ytr = d['y'][:, 0].astype(np.int32)

    d = loadmat(os.path.join(datadir, 'synth_test_32x32.mat'))
    Xts = np.rollaxis(d['X'], -1).astype(np.float32) / 255
    Yts = d['y'][:, 0].astype(np.int32)
    return (Xtr, Ytr), [(Xts, Yts)]

def gtsrb():
    # http://benchmark.ini.rub.de/?section=gtsrb
    datadir = os.path.expanduser('~/.keras/datasets/gtsrb')
    if not os.path.exists(datadir):
        url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
        tf.keras.utils.get_file('GTSRB-Training_fixed.zip',
            url + 'GTSRB-Training_fixed.zip', cache_subdir=datadir, extract=True)
        tf.keras.utils.get_file('GTSRB_Final_Test_Images.zip',
            url + 'GTSRB_Final_Test_Images.zip', cache_subdir=datadir, extract=True)
        tf.keras.utils.get_file('GTSRB_Final_Test_GT.zip',
            url + 'GTSRB_Final_Test_GT.zip', cache_subdir=datadir, extract=True)

    # test average size = (50, 50)
    subdir = os.path.join(datadir, 'GTSRB', 'Training')
    dirs = [d for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
    files = [[os.path.join(subdir, d, f) for f in os.listdir(os.path.join(subdir, d))
        if f.endswith('.ppm')] for d in dirs]
    assert len(files) == len(dirs)
    Xtr = np.array([resize(imread(f), (64, 64)) for fs in files for f in fs], np.float32)
    Ytr = np.array([int(d) for d, fs in zip(dirs, files) for _ in fs], np.int32)

    subdir = 'GTSRB/Final_Test/Images'
    files = sorted(os.listdir(os.path.join(datadir, subdir)))
    files = [f for f in files if f.endswith('.ppm')]
    Xts = [resize(imread(os.path.join(datadir, subdir, f)), (64, 64)) for f in files]
    Xts = np.array(Xts, np.float32)
    txt = os.path.join(datadir, 'GT-final_test.csv')
    Yts = np.loadtxt(txt, np.int32, delimiter=';', skiprows=1, usecols=[-1])
    return (Xtr, Ytr), [(Xts, Yts)]

def synsigns(split=True):
    # http://graphics.cs.msu.ru/en/node/1337
    # (it's possible this is constructed from the previous dataset)
    datadir = os.path.expanduser('~/.keras/datasets/synsigns')
    if not os.path.exists(datadir):
        url = 'https://www.dropbox.com/s/7izi9lccg163on1/synthetic_data.zip?dl=1'
        tf.keras.utils.get_file('synthetic_data.zip',
            'https://www.dropbox.com/s/7izi9lccg163on1/synthetic_data.zip?dl=1',
            cache_subdir=datadir, extract=True)

    subdir = os.path.join(datadir, 'synthetic_data')
    txt = os.path.join(subdir, 'train_labelling.txt')
    files = [os.path.join(subdir, f.split()[0]) for f in open(txt)]
    # all image sizes are (40, 40)
    X = [resize(imread(f), (64, 64)).astype(np.float32) for f in files]
    X = np.array(X)
    Y = np.loadtxt(txt, np.int32, usecols=[1])
    if split:
        rand = np.random.RandomState(0)
        ix = np.random.choice(len(X), len(X), False)
        i = int(len(X)*0.75)
        return (X[ix[:i]], Y[ix[:i]]), [(X[ix[i:]], Y[ix[i:]])]
    return np.array(X), Y

def dogsvscatsredux(split=True):
    # https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
    datadir = os.path.expanduser('~/.keras/datasets/dogsvscatsredux')
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
        os.system(f'kaggle competitions download dogs-vs-cats-redux-kernels-edition -p {datadir}')
        os.system(f'unzip {datadir}/dogs-vs-cats-redux-kernels-edition.zip -d {datadir}')
        os.system(f'unzip {datadir}/train.zip -d {datadir}')
    subdir = os.path.join(datadir, 'train')
    classes = os.listdir(subdir)
    files = [os.path.join(subdir, f) for f in classes]
    X = [resize(imread(f), (64, 64)).astype(np.float32) for f in files]
    X = np.array(X)
    Y = np.array([c.startswith('dog') for c in classes], np.int32)
    if split:
        rand = np.random.RandomState(0)
        ix = np.random.choice(len(X), len(X), False)
        i = int(len(X)*0.75)
        return (X[ix[:i]], Y[ix[:i]]), [(X[ix[i:]], Y[ix[i:]])]
    return X, Y

# cifar10 and stl10 need some class normalization
# we remove frog (cifar10) and monkey (stl10) and normalize the rest
#
# cifar10
# airplane, automobile, bird,      cat, deer, dog, (frog), horse,          ship, truck
# stl10
# airplane,             bird, car, cat, deer, dog,        horse, (monkey), ship, truck

def cifar10_to_stl10(X, Y):
    X = X[Y != 6]  # remove frog
    Y = Y[Y != 6]
    Y[Y > 6] = Y[Y > 6]-1
    return X, Y

def stl10_to_cifar10(X, Y):
    X = X[Y != 7]  # remove monkey
    Y = Y[Y != 7]
    Y[Y > 7] = Y[Y > 7]-1
    bird = Y == 1
    car = Y == 2
    Y[car] = 1
    Y[bird] = 2
    return X, Y

def cifar10_stl10():
    (Xtr, Ytr), [(Xts1, Yts1)] = cifar10()
    Xtr, Ytr = cifar10_to_stl10(Xtr, Ytr)
    Xts1, Yts1 = cifar10_to_stl10(Xts1, Yts1)
    _, [(Xts2, Yts2)] = stl10()
    Xts2, Yts2 = stl10_to_cifar10(Xts2, Yts2)
    return (Xtr, Ytr), [(Xts1, Yts1), (Xts2, Yts2)]

def stl10_cifar10():
    (Xtr, Ytr), [(Xts1, Yts1)] = stl10()
    Xtr, Ytr = stl10_to_cifar10(Xtr, Ytr)
    Xts1, Yts1 = stl10_to_cifar10(Xts1, Yts1)
    _, [(Xts2, Yts2)] = cifar10()
    Xts2, Yts2 = cifar10_to_stl10(Xts2, Yts2)
    return (Xtr, Ytr), [(Xts1, Yts1), (Xts2, Yts2)]

def gtsrb_synsigns():
    tr, ts1 = gtsrb()
    ts2 = synsigns(False)
    return tr, ts1+[ts2]

def svhn_synthdigits():
    tr, ts1 = svhn()
    _, ts2 = synthdigits()
    return tr, ts1+ts2

def synthdigits_svhn():
    tr, ts1 = synthdigits()
    _, ts2 = svhn()
    return tr, ts1+ts2

el = disk(2)[..., np.newaxis]

def apply_background(x, b):
    mask = np.logical_not(binary_dilation(x > 0, el))
    x[mask] = b[mask]

def apply_all_backgrounds(X, Y):
    res = []
    for f in mybackgrounds.l:
        b = f(X.shape[1])[..., np.newaxis]
        X_ = X.copy()
        for x in X_:
            apply_background(x, b)
        res.append((X_, Y))
    return res

def apply_noises(X, Y):
    res = []
    for rate in np.concatenate((np.arange(0, 0.1, 0.01), np.arange(0.1, 1+0.1, 0.1))):
        b = mybackgrounds.salt_pepper(X.shape[1], rate)[..., np.newaxis]
        X_ = X.copy()
        for x in X_:
            apply_background(x, b)
        res.append((X_, Y))
    return res

def mnist_backgrounds():
    tr, ts = mnist()
    return tr, ts + apply_all_backgrounds(*ts[0])

def fashionmnist_backgrounds():
    tr, ts = fashionmnist()
    return tr, ts + apply_all_backgrounds(*ts[0])

def mnist_noises():
    tr, ts = mnist()
    return tr, apply_noises(*ts[0])

def fashionmnist_noises():
    tr, ts = fashionmnist()
    return tr, apply_noises(*ts[0])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--imsave', action='store_true')
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from skimage.io import imsave
    tr, ts = globals()[args.dataset]()
    d = [tr] + ts
    for j, (X, Y) in enumerate(d):
        print('X:', X.shape, X.min(), X.max(), X.dtype)
        print('Y:', Y.shape, Y.min(), Y.max(), Y.dtype)
        for i, (x, y) in enumerate(zip(X[:6], Y[:6])):
            plt.subplot(2, 3, i+1)
            bw = x.shape[-1] == 1
            if bw:
                plt.imshow(x[..., 0], cmap='gray_r')
            else:
                plt.imshow(x)
            plt.title(str(y))
            if args.imsave:
                imsave(f'dataset-{args.dataset}-{j}-{i}.png', ((1-x if bw else x)*255).astype(np.uint8))
        plt.suptitle(f'{args.dataset} {j+1}/{len(d)}')
        plt.show()
