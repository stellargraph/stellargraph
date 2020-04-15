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
from tensorflow.keras.utils import Sequence

from . import (
    Generator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    DirectedGraphSAGENodeGenerator,
    FullBatchSequence,
    SparseFullBatchSequence,
    NodeSequence,
)

class CorruptedGenerator(Generator):
    """
    Keras compatible data generator that wraps :class: `FullBatchNodeGenerator` and provides corrupted
    data for training Deep Graph Infomax.

    Args:
        base_generator: the uncorrupted Sequence object.
    """

    def __init__(self, base_generator):

        if not isinstance(
            base_generator,
            (
                FullBatchNodeGenerator,
                GraphSAGENodeGenerator,
                DirectedGraphSAGENodeGenerator,
            ),
        ):
            raise TypeError(
                f"base_generator: expected FullBatchNodeGenerator, GraphSAGENodeGenerator, "
                f"or DirectedGraphSAGENodeGenerator, found {type(base_generator).__name__}"
            )
        self.base_generator = base_generator

    def flow(self, *args, **kwargs):
        """
        Creates the corrupted :class: `Sequence` object for training Deep Graph Infomax.

        Args:
            args: the positional arguments for the self.base_generator.flow(...) method
            kwargs: the keyword arguments for the self.base_generator.flow(...) method
        """
        return CorruptedNodeSequence(self.base_generator.flow(*args, **kwargs))

class CorruptedNodeSequence(Sequence):
    """
    Keras compatible data generator that wraps a FullBatchSequence ot SparseFullBatchSequence and provides corrupted
    data for training Deep Graph Infomax.

    Args:
        base_generator: the uncorrupted Sequence object.
    """

    def __init__(self, base_generator):

        self.base_generator = base_generator

        if isinstance(base_generator, (FullBatchSequence, SparseFullBatchSequence)):
            self.targets = np.tile(
                [1.0, 0.0], reps=(1, len(base_generator.target_indices), 1),
            )
        elif isinstance(base_generator, NodeSequence):
            self.targets = np.tile([1.0, 0.0], reps=(base_generator.batch_size, 1))
        else:
            raise TypeError(
                f"base_generator: expected FullBatchSequence, SparseFullBatchSequence, "
                f"or NodeSequence, found {type(base_generator).__name__}"
            )

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, index):

        inputs, _ = self.base_generator[index]

        if isinstance(
            self.base_generator, (FullBatchSequence, SparseFullBatchSequence)
        ):

            features = inputs[0]
            shuffled_idxs = np.random.permutation(features.shape[1])
            shuffled_feats = [features[:, shuffled_idxs, :]]
            targets = self.targets

        else:

            features = inputs
            feature_dim = features[0].shape[-1]
            head_nodes = features[0].shape[0]

            shuffled_feats = np.concatenate(
                [x.reshape(-1, feature_dim) for x in features], axis=0,
            )

            np.random.shuffle(shuffled_feats)
            shuffled_feats = shuffled_feats.reshape((head_nodes, -1, feature_dim))
            shuffled_feats = np.split(
                shuffled_feats, np.cumsum([y.shape[1] for y in features])[:-1], axis=1
            )

            targets = self.targets[:head_nodes, :]

        return shuffled_feats + inputs, targets
    
