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


_sg_seed = None


def get_seed(seed):
    """
    Convenience function to use the global seed by default if the provided seed is None.

    Args:
        seed (int, optional): seed value

    Returns:
        seed if not None, otherwise the global seed

    """
    if seed is None:
        return _sg_seed
    else:
        return seed


def set_seed(seed, set_np_seed=True, set_tf_seed=True, set_random_seed=True):
    """
    Set the seed for all possible randomness in StellarGraph. Note that this
    also sets the global random seed for the following external modules:
        * numpy
        * tensorflow
        * random

    Args:
        seed (int, optional): seed value
        set_np_seed (bool, default True): If true, mutate the global numpy seed
        set_tf_seed (bool, default True): If true, mutate the global tensorflow seed
        set_random_seed (bool, default True): If true, mutate the global random module seed

    """
    global _sg_seed
    _sg_seed = seed

    if set_np_seed:
        np.random.seed(seed)
    if set_tf_seed:
        tf.random.set_seed(seed)
    if set_random_seed:
        random.seed(seed)
