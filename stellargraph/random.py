# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
``stellargraph.random`` contains functions to control the randomness behaviour in StellarGraph.

"""
# `random_state` is not user-facing
__all__ = ["set_seed"]

import random as rn
import numpy.random as np_rn
import threading
from collections import namedtuple


RandomState = namedtuple("RandomState", "random, numpy")


def _global_state():
    return RandomState(rn, np_rn)


def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))


_rs = _global_state()


def random_state(seed):
    """
    Create a RandomState using the provided seed. If seed is None, return the global RandomState.

    Args:
        seed (int, optional): random seed

    Returns:
        RandomState object
    """
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)


def set_seed(seed):
    """
    Create a new global RandomState using the provided seed. If seed is None, StellarGraph's global
    RandomState object simply wraps the global random state for each external module.

    When trying to create a reproducible workflow using this function, please note that this seed
    only controls the randomness of the non-tensorflow part of the library. Randomness within
    Tensorflow layers is controlled via Tensorflow's own global random seed, which can be set using
    ``tensorflow.random.set_seed``.

    Args:
        seed (int, optional): random seed

    """
    global _rs
    if seed is None:
        _rs = _global_state()
    else:
        _rs = _seeded_state(seed)


class SeededPerBatch:
    """
    Internal utility class for managing a random state per batch number in a multi-threaded
    environment.

    """

    def __init__(self, create_with_seed, seed):
        self._create_with_seed = create_with_seed
        self._walkers = []
        self._lock = threading.Lock()
        self._rs, _ = random_state(seed)

    def __getitem__(self, batch_num):
        self._lock.acquire()
        try:
            return self._walkers[batch_num]
        except IndexError:
            # always create a new seeded sampler in ascending order of batch number
            # this ensures seeds are deterministic even when batches are run in parallel
            self._walkers.extend(
                self._create_with_seed(self._rs.randrange(2 ** 32))
                for _ in range(len(self._walkers), batch_num + 1)
            )
            return self._walkers[batch_num]
        finally:
            self._lock.release()
