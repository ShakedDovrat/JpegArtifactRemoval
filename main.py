import time
import os

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from data_handling import DataGenerator
from models import identity, simplest, unet
from pprint import pprint, pformat


class Config:
    def __init__(self, params):
        self.images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
        self.image_shape = [512, 512, 1]
        curr_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        self.run_output_dir = 'run_output_' + curr_time
        self.trained_model_path = os.path.join(self.run_output_dir, 'best_model.h5')
        self.log_path = os.path.join(self.run_output_dir, 'log.txt')

        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.num_series = 10

        self.batch_size = 4
        self.lr = params.get('lr', 1e-4)
        self.epochs = 50

        self.model_fun = unet#simplest

        print(pformat(vars(self)))
        os.mkdir(self.run_output_dir)
        with open(self.log_path, 'w') as f:
            print(pformat( vars(self)), file=f)


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
        checkpoint = ModelCheckpoint(self.config.trained_model_path,
                                     monitor='val_mean_squared_error',
                                     verbose=1, save_best_only=True)  # , save_weights_only=True)
        callbacks = [checkpoint]

        self.model.fit_generator(self.data_generators['train'], epochs=self.config.epochs,  # steps_per_epoch=train_samples_per_epoch,
                                 validation_data=self.data_generators['val'], callbacks=callbacks)

    def evaluate(self, dataset_name='val'):
        loss_mse, metric_mse = self.model.evaluate_generator(self.data_generators[dataset_name])
        loss_rmse, metric_rmse = np.sqrt(loss_mse), np.sqrt(metric_mse)
        result_str = 'loss_rmse = {}'.format(loss_rmse), 'metric_rmse = = {}'.format(metric_rmse)
        print(result_str)
        with open(self.config.log_path, 'a') as f:
            print(result_str, file=f)

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
