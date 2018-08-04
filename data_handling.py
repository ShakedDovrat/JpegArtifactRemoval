import os
import pickle
import logging

import numpy as np
import keras
from keras.utils import generic_utils
import cv2


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images_dir, series_idxs, batch_size, image_size, data_mean=0.0, load_x_only=False):
        self.images_dir = images_dir
        self.series_idxs = series_idxs
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_mean = data_mean
        self.load_x_only = load_x_only

        all_image_names = list(set([os.path.splitext(name)[0] for name in os.listdir(self.images_dir)]))
        self.image_names = [name for name in all_image_names if int(name.split('_')[1]) in series_idxs]
        self.idxs = np.arange(len(self.image_names))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches"""
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index):
        """Generate a single batch"""
        idxs = self.idxs[index*self.batch_size:(index+1)*self.batch_size]
        x = np.empty((self.batch_size, *self.image_size))
        y = np.empty((self.batch_size, *self.image_size))
        for i, image_idx in enumerate(idxs):
            x[i] = self._load_image(image_idx, '.jpg')
            if self.load_x_only:
                y[i] = None
            else:
                y[i] = self._load_image(image_idx, '.bmp')

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.idxs)

    def _load_image(self, image_idx, extension):
        image_path = os.path.join(self.images_dir, self.image_names[image_idx] + extension)
        return np.expand_dims(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) - self.data_mean, axis=-1)


def find_data_mean(generator, cache_file, use_cache=True):
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info('Calculating data mean')
        # means = [np.mean(x.flatten()) for x, _ in generator]
        means = np.zeros(len(generator))
        progress_bar = generic_utils.Progbar(len(generator))
        for i, (x, _) in enumerate(generator):
            means[i] = np.mean(x.flatten())
            progress_bar.add(1)

        mean = np.mean(means)
        with open(cache_file, 'wb') as f:
            pickle.dump(mean, f)
        return mean


class DataNormalizer:
    def __init__(self, config):
        self.config = config
        self.file_name_format = 'data_mean_series_{0:03d}.pkl'

    def get_data_mean(self, series_idxs):
        means = []
        progress_bar = generic_utils.Progbar(len(series_idxs))
        for series_idx in series_idxs:
            cache_file = os.path.join(self.config.images_dir, self.file_name_format.format(series_idx))
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    curr_mean = pickle.load(f)
            else:
                curr_mean = self._calc_data_mean(series_idx)
                with open(cache_file, 'wb') as f:
                    pickle.dump(curr_mean, f)
            means.append(curr_mean)
            progress_bar.add(1)

        return np.mean(means)

    def _calc_data_mean(self, series_idx):
        logging.info('Calculating data mean for series #{0:03d}'.format(series_idx))
        generator = DataGenerator(self.config.images_dir, (series_idx,), batch_size=self.config.batch_size,
                                  image_size=self.config.image_shape, load_x_only=True)
        mean = np.mean([np.mean(x.flatten()) for x, _ in generator])
        logging.info('Done.')
        return mean
