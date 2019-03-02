import numpy as np
from numpy.linalg import norm
from PIL import Image
from random import randint
from scipy.linalg import eigh
from scipy import misc
from math import pi
import argparse


modules = [1, -1, 1j, -1j]
delta = 1.001
M = 40


def pearson_corr(u, v):
    s = u.dot(v)
    return (norm(s) / (norm(u) * norm(v))) ** 2


def transform(y, delta=1.001):
    return max((y - 1) / (y + np.sqrt(delta) - 1), -M)


def titan(n):
    return np.einsum('i, j->ij', np.arange(n), np.arange(n))


class Diffraction(object):
    def __init__(self, signal, L):
        self.L = L
        self.size = signal.shape
        self.x = np.reshape(signal, -1)
        self.a, self.h = self._setup_sensing(L, *self.size)

    def _setup_sensing(self, L, d1, d2):
        d = d1 * d2
        idx_tab1 = titan(d1)
        idx_tab2 = titan(d2)
        e1 = np.exp(idx_tab1 * 2j * pi / d1)
        e2 = np.exp(idx_tab2 * 2j * pi / d2)
        a = np.einsum('ik, jm->kmij', e1, e2)
        a = np.reshape(a, (d, d))
        h = np.random.choice(modules, size=(L, d))
        return a, h

    def _measure(self, l, k):
        coef = self.a[k] * self.h[l]
        y = norm(coef.dot(self.x)) ** 2
        return transform(y) * np.einsum('i, j->ij', coef, np.conj(coef))

    def _process(self):
        D = 0
        d1, d2 = self.size
        d = d1 * d2
        print('signal size {} {}'.format(d1, d2))
        for l in range(self.L):
            for k in range(d):
                D += self._measure(l, k)
            print('step {}'.format(l), end='\r', flush=True)
        print('\ncompute eigenvector corresponding to largest eigenvalue ...')
        w, v = eigh(D / (self.L * d) + 100 * np.identity(d, dtype=complex)) # 100 * np.identity(d, dtype=complex))
        m = max(range(d), key=lambda i: np.sign(w[i].real) * norm(w[i]))
        return v


def main(args):
    im_path = os.path.join('img', args.filename)

    size = [args.size] * 2

    im = Image.open(im_path)
    im = im.resize(size)
    im_ = im.resize((400, 400))
    im_.show()
    im = np.array(im)
    m_x = [np.max(im[:, :, i]) for i in range(3)]
    print(im.shape)
    #x = np.random.uniform(size=size)
    ori = np.zeros(shape=(*size, 3))
    for i in range(3):
        x = im[:, :, i] / 255.
        #x = 4 * x / norm(x)
        x_ = np.reshape(x, -1)

        model = Diffraction(x, args.delta)
        v = model._process()
        m = max(range(v.shape[1]), key=lambda i: pearson_corr(v[:, i], x_))
        sol = np.abs(v[:, m].real)
        print(pearson_corr(sol, x_))
        ori[:, :, i] = np.reshape(sol / np.max(sol) * m_x[i], size)

    img = Image.fromarray(ori.astype('uint8'))
    img = img.resize((400, 400))
    img.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', help='image_path', nargs='?', default='test.jpg')
    parser.add_argument('--delta', '-d', help='delta value (integer part)', nargs='?', default=8, type=int)
    parser.add_argument('--size', '-s', help='image size (only square images)', nargs='?', default=20, type=int)
    args = parser.parse_args()
    main(args)
