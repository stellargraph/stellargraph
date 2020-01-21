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
    if seed is None:
        return _sg_seed
    else:
        return seed


def set_seed(seed):
    """
    Set the seed for all possible randomness in StellarGraph

    Args:
        s (int, optional): seed value

    """
    global _sg_seed
    _sg_seed = seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
