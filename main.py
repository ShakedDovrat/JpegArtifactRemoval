import time
import os

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from data_handling import DataGenerator
from models import identity, simplest, unet


class Config:
    def __init__(self):
        self.seed = 0
        self.images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
        self.image_shape = [512, 512, 1]
        curr_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        self.run_output_dir = 'run_output_' + curr_time
        self.trained_model_path = os.path.join(self.run_output_dir, 'best_model.h5')

        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.num_series = 10

        self.batch_size = 4
        self.lr = 1e-4
        self.epochs = 50

        self.model_fun = unet#simplest


class Model:
    def __init__(self, config):
        self.config = config

        np.random.seed(config.seed)

        os.mkdir(config.run_output_dir)
        self.datasets_series_idxs = self._separate_train_val_test()
        self.data_generators = self._get_data_generators()
        self.model = self._build_model()

    def run(self):
        self.train()
        self.model = load_model(self.config.trained_model_path)
        self.evaluate()

    def train(self):
        checkpoint = ModelCheckpoint(self.config.trained_model_path,
                                     monitor='val_mean_squared_error',
                                     verbose=1, save_best_only=True)  # , save_weights_only=True)
        callbacks = [checkpoint]

        self.model.fit_generator(self.data_generators['train'], epochs=self.config.epochs,  # steps_per_epoch=train_samples_per_epoch,
                                 validation_data=self.data_generators['val'], callbacks=callbacks)

    def evaluate(self, dataset_name='val'):
        loss_mse, metric_mse = self.model.evaluate_generator(self.data_generators[dataset_name])
        loss_rmse, metric_rmse = np.sqrt(loss_mse), np.sqrt(metric_mse)
        print('loss_rmse = {}'.format(loss_rmse), 'metric_rmse = = {}'.format(metric_rmse))

    def _build_model(self):
        model = self.config.model_fun(self.config.image_shape)
        model.compile(optimizer=Adam(lr=self.config.lr), loss='mean_squared_error', metrics=['mean_squared_error'])
        model.summary()
        return model

    def _separate_train_val_test(self):
        num_train_series = int(np.round(self.config.num_series * self.config.train_ratio))
        num_val_series = int(np.round(self.config.num_series * self.config.val_ratio))
        idxs = np.arange(self.config.num_series)
        np.random.shuffle(idxs)
        return {'train': idxs[:num_train_series],
                'val': idxs[num_train_series:num_train_series + num_val_series],
                'test': idxs[num_train_series + num_val_series:]}

    def _get_data_generators(self):
        dataset_names = ('train', 'val', 'test')
        return {name: DataGenerator(self.config.images_dir, self.datasets_series_idxs[name], self.config.batch_size,
                                    image_size=self.config.image_shape)
                for name in dataset_names}


def main():
    model = Model(Config())
    model.run()


if __name__ == '__main__':
    main()
