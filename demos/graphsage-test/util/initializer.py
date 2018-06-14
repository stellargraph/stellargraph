import numpy as np
from keras.initializers import RandomUniform


def glorot_initializer(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0] + shape[1]))
    return RandomUniform(minval=-init_range, maxval=init_range, seed=None)
