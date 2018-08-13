import time
import os
import logging
import sys
from pprint import pformat

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model

from data_handling import DataGenerator, DataNormalizer
from models import identity, simplest, unet, unet_16, srcnn, ar_cnn, dn_cnn_b


class Config:
    def __init__(self, model_dir=None, params={}):
        self.images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
        base_output_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/output'

        # Data params
        self.image_shape = [512, 512, 1]
        self.num_series = 10
        self.train_ratio = 0.6
        self.val_ratio = 0.2

        # Train params
        self.batch_size = 1
        self.lr = params.get('lr', 1e-4)
        self.epochs = 100

        # Model params
        self.model_fun = dn_cnn_b  # srcnn,simplest,unet,unet_16,ar_cnn
        self.is_residual = self.model_fun.__name__.startswith('dn_cnn')

        model_dir = model_dir or 'run_output_' + time.strftime('%Y_%m_%d_%H_%M_%S')
        self.run_output_dir = os.path.join(base_output_dir, model_dir)
        if not os.path.exists(self.run_output_dir):
            os.makedirs(self.run_output_dir)
        self.trained_model_path = os.path.join(self.run_output_dir, 'best_model.h5')
        self._set_logger()
        logging.info('Config = ' + pformat(vars(self)))

    def _set_logger(self):
        log_path = os.path.join(self.run_output_dir, 'log.txt')

        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        file_handler = logging.FileHandler(log_path)
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
        logging.info('Model = ' + pformat(vars(self)))

    def train(self):
        callbacks = self._get_callbacks()

        self.model.fit_generator(self.data_generators['train'], epochs=self.config.epochs,
                                 validation_data=self.data_generators['val'], callbacks=callbacks)

    def test(self):
        self.model = load_model(self.config.trained_model_path)
        self._evaluate()

    def _evaluate(self, dataset_name='val'):
        if self.config.is_residual:
            generator = self.data_generators[dataset_name]
            rmse = np.zeros((self.config.batch_size, len(generator)))
            for i in range(len(generator)):
                x, y = generator[i]
                y_res = self.model.predict(x)
                y_hat = x + y_res
                rmse[:, i] = [np.linalg.norm(diff.flatten(), ord=2) for diff in y_hat - y]
            rmse_total = np.mean(rmse.flatten())
            result_str = '{}: rmse = {}'.format(dataset_name, rmse_total)
        else:
            loss_mse, metric_mse = self.model.evaluate_generator(self.data_generators[dataset_name])
            loss_rmse, metric_rmse = np.sqrt(loss_mse), np.sqrt(metric_mse)
            result_str = '{}: loss_rmse = {}; metric_rmse = {}'.format(dataset_name, loss_rmse, metric_rmse)
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
        normalizer = DataNormalizer(self.config)
        data_mean = normalizer.get_data_mean(self.datasets_series_idxs['train'])

        dataset_names = ('train', 'val', 'test')
        return {name: DataGenerator(self.config.images_dir, self.datasets_series_idxs[name], self.config.batch_size,
                                    image_size=self.config.image_shape, data_mean=data_mean,
                                    is_residual=self.config.is_residual)
                for name in dataset_names}

    def _get_callbacks(self):
        checkpoint = ModelCheckpoint(self.config.trained_model_path, monitor='val_mean_squared_error',
                                     verbose=1, save_best_only=True)  # , save_weights_only=True)
        tensor_board = TensorBoard(self.config.run_output_dir, batch_size=self.config.batch_size)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-8, verbose=1)

        return [checkpoint, tensor_board, reduce_lr, early_stop]


def test():
    model_dir = 'run_output_2018_08_05_18_23_05'
    model = Model(Config(model_dir))
    model.test()


def train(params={}):
    model = Model(Config(params=params))
    model.train()
    model.test()


def params_search():
    seed = 1
    num_runs = 20

    np.random.seed(seed)

    for idx in range(num_runs):
        lr = 10 ** np.random.uniform(-6, -2)

        print('Run #{}, lr={}'.format(idx, lr))
        train({'lr': lr})


if __name__ == '__main__':
    train()
    # test()
    # params_search()
