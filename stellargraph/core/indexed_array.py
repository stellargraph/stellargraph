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

import numpy as np

from .utils import zero_sized_array


class IndexedArray:
    """
    An array where the first dimension is indexed.

    This is a reduced Pandas DataFrame. It has:

    - multidimensional data support, where each element ``values[idx, ...]`` can be a vector, matrix or
      even higher rank object

    - a requirement that all values have the same type

    - labels for the elements of the first axis e.g. ``index[0]`` is the label for the ``values[0,
      ...]`` element.

    - no labels for other axes

    - less overhead (but less API) than a Pandas DataFrame

    Args:
        values (numpy.ndarray, optional): an array of rank at least 2 of data, where the first axis
            is indexed.

        index (sequence, optional): a sequence of labels or IDs, one for each element of the first
            axis. If not specified, this defaults to sequential integers starting at 0
    """

    def __init__(self, values=None, index=None):
        def index_len():
            # compute the length of the index, intercepting an error to provide a better message
            try:
                return len(index)
            except:
                raise TypeError(
                    f"index: expected a sequence (with a '__len__' method), found {type(index).__name__}"
                )

        if values is None:
            if index is None:
                index = range(0)

            # uint8 is essentially maximally promotable
            values = zero_sized_array((index_len(), 0), dtype=np.uint8)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"values: expected a NumPy array for the features, found {type(values).__name__}"
            )

        if len(values.shape) < 2:
            raise ValueError(
                f"values: expected an array with shape length >= 2, found shape {values.shape} of length {len(values.shape)}"
            )

        values_len = values.shape[0]

        if index is None:
            index = range(values_len)

        if values_len != index_len():
            raise ValueError(
                f"values: expected the index length {index_len()} to match the first dimension of values, found {values_len} rows"
            )

        self.index = index
        self.values = values
