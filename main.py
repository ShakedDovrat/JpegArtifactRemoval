import numpy as np

from data_handling import DataGenerator


def separate_train_val_test(train_ratio=0.6, val_ratio=0.2, num_series=10):
    num_train_series = int(np.round(num_series * train_ratio))
    num_val_series = int(np.round(num_series * val_ratio))
    idxs = np.arange(num_series)
    np.random.shuffle(idxs)
    return {'train': idxs[:num_train_series],
            'val': idxs[num_train_series:num_train_series + num_val_series],
            'test': idxs[num_train_series + num_val_series:]}


def main():
    images_dir = '/media/almond/magnetic-2TB/science/viz-ai-exercise/data/takehome'
    datasets_series_idxs = separate_train_val_test()
    generator = DataGenerator(images_dir, datasets_series_idxs['train'], 32)
    x, y = generator[0]
    pass


if __name__ == '__main__':
    main()
