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

__all__ = [
    "SlidingFeaturesNodeGenerator",
    "SlidingFeaturesNodeSequence",
]

import numpy as np
from . import Generator
from tensorflow.keras.utils import Sequence

from ..core.validation import require_integer_in_range


class SlidingFeaturesNodeGenerator(Generator):
    """
    A data generator for a graph containing sequence data, created by sliding windows across the
    features of each node in a graph.

    Args:
        G (StellarGraph): a graph instance where the node features are ordered sequence data
        window_size (int): the number of sequence points included in the sliding window.
        batch_size (int, optional): the number of sliding windows to include in each batch.
    """

    def __init__(self, G, window_size, batch_size=1):
        require_integer_in_range(window_size, "window_size", min_val=1)
        require_integer_in_range(batch_size, "batch_size", min_val=1)

        self.graph = G

        node_type = G.unique_node_type(
            "G: expected a graph with a single node type, found a graph with node types: %(found)s"
        )
        self._features = G.node_features(node_type=node_type)
        if len(self._features.shape) == 3:
            self.variates = self._features.shape[2]
        else:
            self.variates = None

        self.window_size = window_size
        self._batch_size = batch_size

    def num_batch_dims(self):
        return 1

    def flow(self, sequence_iloc_slice, target_distance=None):
        """
        Create a sequence object for time series prediction within the given section of the node
        features.

        This handles both univariate data (each node has a single associated feature vector) and
        multivariate data (each node has an associated feature tensor). The features are always
        sliced and indexed along the first feature axis.

        Args:
            sequence_iloc_slice (slice):
                A slice object of the range of features from which to select windows. A slice object
                is the object form of ``:`` within ``[...]``, e.g. ``slice(a, b)`` is equivalent to
                the ``a:b`` in ``v[a:b]``, and ``slice(None, b)`` is equivalent to ``v[:b]``. As
                with that slicing, this parameter is inclusive in the start and exclusive in the
                end.

                For example, suppose the graph has feature vectors of length 10 and ``window_size =
                3``:

                * passing in ``slice(None, None)`` will create 7 windows across all 10 features
                  starting with the features slice ``0:3``, then ``1:4``, and so on.

                * passing in ``slice(4, 7)`` will create just one window, slicing the three elements
                  ``4:7``.

                For training, one might do a train-test split by choosing a boundary and considering
                everything before that as training data, and everything after, e.g. 80% of the
                features::

                    train_end = int(0.8 * sequence_length)
                    train_gen = sliding_generator.flow(slice(None, train_end))
                    test_gen = sliding_generator.flow(slice(train_end, None))

            target_distance (int, optional):
                The distance from the last element of each window to select an element to include as
                a supervised training target. Note: this always stays within the slice defined by
                ``sequence_iloc_slice``.

                Continuing the example above: a call like ``sliding_generator.flow(slice(4, 9),
                target_distance=1)`` will yield two pairs of window and target:

                * a feature window slicing ``4:7`` which includes the features at indices 4, 5, 6,
                  and then a target feature at index 7 (distance 1 from the last element of the
                  feature window)

                * a feature window slicing ``5:8`` and a target feature from index 8.

        Returns:
            A Keras sequence that yields batches of sliced windows of features, and, optionally,
            selected target values.
        """
        return SlidingFeaturesNodeSequence(
            self._features,
            self.window_size,
            self._batch_size,
            sequence_iloc_slice,
            target_distance,
        )


class SlidingFeaturesNodeSequence(Sequence):
    def __init__(
        self, features, window_size, batch_size, sequence_iloc_slice, target_distance
    ):
        if target_distance is not None:
            require_integer_in_range(target_distance, "target_distance", min_val=1)

        if not isinstance(sequence_iloc_slice, slice):
            raise TypeError(
                f"sequence_iloc_slice: expected a slice(...) object, found {type(sequence_iloc_slice).__name__}"
            )

        if sequence_iloc_slice.step not in (None, 1):
            raise TypeError(
                f"sequence_iloc_slice: expected a slice object with a step = 1, found step = {sequence_iloc_slice.step}"
            )

        self._features = features[:, sequence_iloc_slice, ...]
        shape = self._features.shape
        self._num_nodes = shape[0]
        self._num_sequence_samples = shape[1]
        self._num_sequence_variates = shape[2:]

        self._window_size = window_size
        self._target_distance = target_distance
        self._batch_size = batch_size

        query_length = window_size + (0 if target_distance is None else target_distance)
        self._num_windows = self._num_sequence_samples - query_length + 1

        # if there's not enough data to fill one window, there's a problem!
        if self._num_windows <= 0:
            if target_distance is None:
                target_str = ""
            else:
                target_str = f" + target_distance={target_distance}"

            total_sequence_samples = features.shape[1]
            start, stop, step = sequence_iloc_slice.indices(total_sequence_samples)
            # non-trivial steps aren't supported at the moment, so this doesn't need to be included
            # in the message
            assert step == 1

            raise ValueError(
                f"expected at least one sliding window of features, found a total window of size {query_length} (window_size={window_size}{target_str}) which is larger than the {self._num_sequence_samples} selected feature sample(s) (sequence_iloc_slice selected from {start} to {stop} in the sequence axis of length {total_sequence_samples})"
            )

    def __len__(self):
        return int(np.ceil(self._num_windows / self._batch_size))

    def __getitem__(self, batch_num):
        first_start = batch_num * self._batch_size
        last_start = min((batch_num + 1) * self._batch_size, self._num_windows)

        has_targets = self._target_distance is not None

        arrays = []
        targets = [] if has_targets else None
        for start in range(first_start, last_start):
            end = start + self._window_size
            arrays.append(self._features[:, start:end, ...])
            if has_targets:
                target_idx = end + self._target_distance - 1
                targets.append(self._features[:, target_idx, ...])

        this_batch_size = last_start - first_start

        batch_feats = np.stack(arrays)
        assert (
            batch_feats.shape
            == (this_batch_size, self._num_nodes, self._window_size)
            + self._num_sequence_variates
        )

        if has_targets:
            batch_targets = np.stack(targets)
            assert (
                batch_targets.shape
                == (this_batch_size, self._num_nodes) + self._num_sequence_variates
            )
        else:
            batch_targets = None

        return [batch_feats], batch_targets
