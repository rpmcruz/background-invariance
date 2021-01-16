import numpy as np
from skimage.draw import circle_perimeter, rectangle

def stripes(size):
    thickness = 2
    interspace = 8
    m = 1
    lim_b = [-size, size] if m > 0 else [0, size*2]
    first_b = lim_b[0]+1
    img = np.zeros((size, size), np.float32)
    for bi in range(first_b, lim_b[1], interspace+thickness):
        for b in range(bi, bi+thickness):
            for x in range(size):
                y = m*x + b
                if 0 <= y < size:
                    img[y, x] = 1
    return img

def checkerboard(size):
    square = 6
    color = 0
    img = np.zeros((size, size), np.float32)
    xi = yi = -2
    for y in range(yi, size, square):
        color_i = color
        for x in range(xi, size, square):
            img[max(0, y):min(size, y+square), max(0, x):min(size, x+square)] = color
            color = 1-color
        color = 1-color_i
    return img

def border(size):
    thickness = 2
    img = np.zeros((size, size), np.float32)
    img[:thickness, :] = 1
    img[size-thickness:, :] = 1
    img[:, :thickness] = 1
    img[:, size-thickness:] = 1
    return img

def circumferences(size):
    img = np.zeros((size, size), np.float32)
    shift = 8
    radius = 6
    rr, cc = circle_perimeter(shift, shift, radius)
    img[rr, cc] = 1
    rr, cc = circle_perimeter(size-shift, shift, radius)
    img[rr, cc] = 1
    rr, cc = circle_perimeter(shift, size-shift, radius)
    img[rr, cc] = 1
    rr, cc = circle_perimeter(size-shift, size-shift, radius)
    img[rr, cc] = 1
    return img

def clock(size):
    img = np.zeros((size, size), np.float32)
    rr, cc = circle_perimeter(size//2, size//2, size//2-1)
    img[rr, cc] = 1
    rr, cc = circle_perimeter(size//2, size//2, size//2-2)
    img[rr, cc] = 1
    w = 2
    h = 6
    rr, cc = rectangle((size//2-w//2, 0), (size//2+w//2, h))
    img[rr, cc] = 1
    rr, cc = rectangle((size//2-w//2, size-h), (size//2+w//2, size-1))
    img[rr, cc] = 1
    rr, cc = rectangle((0, size//2-w//2), (h, size//2+w//2))
    img[rr, cc] = 1
    rr, cc = rectangle((size-h, size//2-w//2), (size-1, size//2+w//2))
    img[rr, cc] = 1
    return img

def salt_pepper(size, rate=0.25):
    img = np.zeros(size*size, np.float32)
    rand = np.random.RandomState(0)
    img[rand.choice(size*size, int(size*size*rate), False)] = 1
    return img.reshape((size, size))

l = [stripes, checkerboard, border, circumferences, clock, salt_pepper]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for i, f in enumerate(l):
        plt.subplot(2, 3, i+1)
        plt.imshow(f(28), cmap='gray', vmin=0, vmax=1)
        plt.title(f.__name__)
        plt.axis('off')
    plt.show()
