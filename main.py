import time
import os

import numpy as np
from keras.metrics import mean_squared_error
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

from data_handling import DataGenerator
from metrics import rmse
from models import identity, simplest, unet


def separate_train_val_test(train_ratio=0.6, val_ratio=0.2, num_series=10):
    num_train_series = int(np.round(num_series * train_ratio))
    num_val_series = int(np.round(num_series * val_ratio))
    idxs = np.arange(num_series)
    np.random.shuffle(idxs)
    return {'train': idxs[:num_train_series],
            'val': idxs[num_train_series:num_train_series + num_val_series],
            'test': idxs[num_train_series + num_val_series:]}


def evaluate_model(model, generator):
    loss_mse, metric_mse = model.evaluate_generator(generator)
    loss_rmse, metric_rmse = np.sqrt(loss_mse), np.sqrt(metric_mse)
    print('loss_rmse = {}'.format(loss_rmse), 'metric_rmse = = {}'.format(metric_rmse))


def main():
    seed = 1337
    images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
    image_shape = [512, 512, 1]
    curr_time = time.strftime('_%d_%m_%Y_%H_%M')
    run_output_dir = 'run_output' + curr_time
    os.mkdir(run_output_dir)

    batch_size = 16
    lr = 1e-4
    epochs = 50

    np.random.seed(seed)
    datasets_series_idxs = separate_train_val_test()

    train_generator = DataGenerator(images_dir, datasets_series_idxs['train'], batch_size, image_size=image_shape)
    val_generator = DataGenerator(images_dir, datasets_series_idxs['val'], batch_size, image_size=image_shape)

    checkpoint = ModelCheckpoint(os.path.join(run_output_dir, 'best_weights.h5'),
                                 monitor='val_mean_squared_error',
                                 verbose=1, save_best_only=True, save_weights_only=True)
    callbacks = [checkpoint]

    # model = identity(image_shape)
    model = simplest(image_shape)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=[mean_squared_error])
    model.summary()

    model.fit_generator(train_generator, epochs=epochs, #steps_per_epoch=train_samples_per_epoch,
                        validation_data=val_generator, callbacks=callbacks)

    evaluate_model(model, val_generator)

    pass


if __name__ == '__main__':
    main()

    # x, y = generator[0]
    # for x,y in generator:
    #     metric = rmse(x, y)
    # rmse_total = np.mean(metric)
