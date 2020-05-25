"""
Data Handler for working with Autoencoders

Author: Simon Thomas
DateL 22-May-2020

"""
import os
import numpy as np
import skimage.io as io

from skimage.transform import resize
from skimage.filters import gaussian


class DataGen:
    """
    A data generator that can handing multiple workers if used in conjunction with
    the model.fit_generator method.

    Enables simplification of images by using pixelation e.g. 128x128 >> 16x16 >> 128x128,
    paired with smooth transitions between scales using alpha, as well as gaussian blurring
    controlled by sigma.

    It is a generator so `X, y = next(generator)` will get a new batch, but indexing also works and
    is preferred e.g. `X, y = generator[0]`.

    """

    def __init__(self, img_dir, img_dim, batch_size, pixel=False, alpha=-1, sigma=False, shuffle=True, style=False):
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.pixel = pixel
        self.alpha = alpha
        self.sigma = sigma
        self.style = style
        self.files = np.array([os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)])
        self.n = len(self.files)
        self.indices = np.arange(0, self.n)
        if shuffle:
            np.random.shuffle(self.indices)
        self.pos = 0

    def _get_indices(self, i):
        return self.indices[i * self.batch_size:i * self.batch_size + self.batch_size]

    def __getitem__(self, i):
        batch = []
        labels = []
        files = self.files[self._get_indices(i)]
        for file in files:
            img = io.imread(file) / 255.
            if img.shape[0] != self.img_dim:
                img = resize(img, (self.img_dim, self.img_dim))

            if self.sigma:
                img = gaussian(img, self.sigma)

            elif self.pixel:
                img_big = resize(img, (self.pixel, self.pixel))
                img_big = resize(img_big, (self.img_dim, self.img_dim), order=0) # nearest

                # if alpha
                if type(self.alpha) != bool and self.alpha >= 0:
                    img_small = resize(img, (self.pixel // 2, self.pixel // 2))
                    img_small = resize(img_small, (self.img_dim, self.img_dim), order=0) # nearest

                    # interpolate
                    img = self.alpha * img_big + (1 - self.alpha) * img_small

                else:
                    img = img_big

            batch.append(img)

        if self.style:
            return (np.stack(batch), np.random.normal(0, 1, (self.batch_size, self.img_dim, self.img_dim, 1))), np.stack(batch)

        return np.stack(batch), np.stack(batch)

    def __next__(self):
        self.pos += 1
        if self.pos > self.n // self.batch_size:
            self.pos = 1
        return self.__getitem__(self.pos - 1)