import numpy as np


def rmse(x, y):
    return np.linalg.norm((x - y).flatten(), ord=2)
