import tensorflow as tf
import numpy as np

def classifier(input_shape, output_shape):
    x = input_layer = tf.keras.layers.Input(input_shape)
    n = int(np.log2(input_shape[0] / 4))
    for i in range(n):
        # vgg block
        filters = 2**(i+5)
        x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape)(x)
    model = tf.keras.models.Model(input_layer, x)
    opt = tf.keras.optimizers.Adam(1e-3)
    return model, opt

def vgg19(input_shape, output_shape):
    vgg19 = tf.keras.applications.VGG19(False)
    x = input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Lambda(lambda x: x[..., ::-1])(x)
    x = tf.keras.layers.Lambda(lambda x:
        x*255 - tf.constant([103.939, 116.779, 123.68], shape=(1, 1, 1, 3)))(x)
    x = vgg19(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape)(x)
    model = tf.keras.models.Model(input_layer, x)
    opt = tf.keras.optimizers.Adam(1e-5)
    return model, opt

def unet(input_shape):
    x = input_layer = tf.keras.layers.Input(input_shape)
    down_layers = []
    n = int(np.log2(input_shape[0] / 4))
    for i in range(n):
        filters = 2**(i+5)
        x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation='relu')(x)
        down_layers.append(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(2**(n-1+5), 3, 1, 'same', activation='relu')(x)
    for i, prev in enumerate(down_layers[::-1]):
        filters = 2**(n-1-i+5)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation='relu')(x)
        x = tf.keras.layers.Concatenate()([x, prev])
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = tf.keras.models.Model(input_layer, x)
    opt = tf.optimizers.Adam(1e-5)
    return model, opt

def gen(output_shape, tiles):
    assert (output_shape[0] / (tiles*2)) % 2 == 0
    def gcd2(a):
        return 1+gcd2(a//2) if a%2 == 0 else 0
    output_shape = (output_shape[0]//tiles, output_shape[1]//tiles, output_shape[2])
    input_dim = max(output_shape[0]//(2**gcd2(output_shape[0])), 4)
    input_shape = (input_dim, input_dim, output_shape[2])
    act = tf.keras.layers.LeakyReLU(0.2)
    x = input_layer = tf.keras.layers.Input([np.prod(input_shape)])
    x = tf.keras.layers.Reshape(input_shape)(x)
    n = int(np.log2(output_shape[0]/input_shape[0]))
    filters = 2**(n+5)
    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation=act)(x)
    for i in range(n):
        filters = 2**(n-1-i+5)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation=act)(x)
    x = tf.keras.layers.Conv2D(output_shape[2], 1, activation='sigmoid')(x)
    model = tf.keras.models.Model(input_layer, x)
    opt = tf.optimizers.Adam(1e-5)
    return model, opt, np.prod(input_shape)
