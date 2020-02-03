# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import tensorflow as tf


_sg_random = random.Random()


def generate_seed(seed=None):
    """
    Convenience function to use the global seed by default if the provided seed is None.

    Args:
        seed (int, optional): seed value

    Returns:
        seed if not None, otherwise the global seed

    """
    if seed is None:
        return _sg_random.randint(0, 2 ** 32 - 1)
    else:
        return seed


def set_seed(seed, set_numpy=True, set_tensorflow=True, set_random=True):
    """
    Set the seed for all possible randomness in StellarGraph. Note that this
    also sets the global random seed for the following external modules:
        * numpy
        * tensorflow
        * random

    Args:
        seed (int, optional): seed value
        set_numpy (bool): If true, mutate the global numpy seed
        set_tensorflow (bool): If true, mutate the global tensorflow seed
        set_random (bool): If true, mutate the global random module seed

    """
    global _sg_random
    _sg_random = random.Random(seed)

    if set_numpy:
        np.random.seed(seed)
    if set_tensorflow:
        tf.random.set_seed(seed)
    if set_random:
        random.seed(seed)
