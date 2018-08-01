import os

import numpy as np
import keras
import cv2


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images_dir, series_idxs, batch_size, image_size=(512, 512)):
        self.images_dir = images_dir
        self.series_idxs = series_idxs
        self.batch_size = batch_size
        self.image_size = image_size

        all_image_names = list(set([os.path.splitext(name)[0] for name in os.listdir(self.images_dir)]))
        self.image_names = [name for name in all_image_names if int(name.split('_')[1]) in series_idxs]
        self.indexes = np.arange(len(self.image_names))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches"""
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index):
        """Generate a single batch"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x = np.empty((self.batch_size, *self.image_size))
        y = np.empty((self.batch_size, *self.image_size))
        for i, image_idx in enumerate(indexes):
            x[i] = self._load_image(image_idx, '.jpg')
            y[i] = self._load_image(image_idx, '.bmp')

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def _load_image(self, image_idx, extension):
        image_path = os.path.join(self.images_dir, self.image_names[image_idx] + extension)
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
