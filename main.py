import time
import os
import logging
import sys

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model

from data_handling import DataGenerator, find_data_mean
from models import identity, simplest, unet, unet_16
from pprint import pformat


class Config:
    def __init__(self, params):
        self.images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
        base_output_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/output'

        self.run_output_dir = os.path.join(base_output_dir, 'run_output_' + time.strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(self.run_output_dir)
        self.trained_model_path = os.path.join(self.run_output_dir, 'best_model.h5')
        self.log_path = os.path.join(self.run_output_dir, 'log.txt')

        self.image_shape = [512, 512, 1]
        self.num_series = 10

        self.train_ratio = 0.6
        self.val_ratio = 0.2

        self.batch_size = 8
        self.lr = params.get('lr', 1e-4)
        self.epochs = 50

        self.model_fun = unet_16  # unet  # simplest

        self._set_logger()
        logging.info(pformat(vars(self)))

    def _set_logger(self):
        logger = logging.getLogger()

        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)


class Model:
    def __init__(self, config):
        self.config = config

        self.datasets_series_idxs = self._separate_train_val_test()
        self.data_generators = self._get_data_generators()
        self.model = self._build_model()

    def run(self):
        self.train()
        self.model = load_model(self.config.trained_model_path)
        self.evaluate()

    def train(self):
        callbacks = self._get_callbacks()

        self.model.fit_generator(self.data_generators['train'], epochs=self.config.epochs,  # steps_per_epoch=train_samples_per_epoch,
                                 validation_data=self.data_generators['val'], callbacks=callbacks)

    def evaluate(self, dataset_name='val'):
        loss_mse, metric_mse = self.model.evaluate_generator(self.data_generators[dataset_name])
        loss_rmse, metric_rmse = np.sqrt(loss_mse), np.sqrt(metric_mse)
        result_str = 'loss_rmse = {}'.format(loss_rmse), 'metric_rmse = = {}'.format(metric_rmse)
        logging.info(result_str)

    def _build_model(self):
        model = self.config.model_fun(self.config.image_shape)
        model.compile(optimizer=Adam(lr=self.config.lr), loss='mean_squared_error', metrics=['mean_squared_error'])
        model.summary(print_fn=logging.info)
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
        train_generator = DataGenerator(self.config.images_dir, self.datasets_series_idxs['train'],
                                        self.config.batch_size, image_size=self.config.image_shape)
        data_mean_path = os.path.join(self.config.images_dir,
                                      'data_mean_{}.pkl'.format(self.datasets_series_idxs['train']))
        data_mean = find_data_mean(train_generator, data_mean_path)
        dataset_names = ('train', 'val', 'test')
        return {name: DataGenerator(self.config.images_dir, self.datasets_series_idxs[name], self.config.batch_size,
                                    image_size=self.config.image_shape, data_mean=data_mean)
                for name in dataset_names}

    def _get_callbacks(self):
        checkpoint = ModelCheckpoint(self.config.trained_model_path, monitor='val_mean_squared_error',
                                     verbose=1, save_best_only=True)  # , save_weights_only=True)
        tensor_board = TensorBoard(self.config.run_output_dir, batch_size=self.config.batch_size)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-8, verbose=1)

        return [checkpoint, tensor_board, reduce_lr, early_stop]


def main(params={}):
    model = Model(Config(params))
    model.run()


def main_params_search():
    seed = 1
    num_runs = 20

    np.random.seed(seed)

    for idx in range(num_runs):
        lr = 10 ** np.random.uniform(-6, -2)

        print('Run #{}, lr={}'.format(idx, lr))
        main({'lr': lr})


if __name__ == '__main__':
    main_params_search()
