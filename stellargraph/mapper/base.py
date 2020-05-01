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

import abc


class Generator(abc.ABC):
    """
    A generator supports creating sequences for input into graph machine learning algorithms via the `flow` method.
    """

    @abc.abstractmethod
    def num_batch_dims(self):
        """
        Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).

        For instance, for full batch methods like GCN, the feature has shape ``1 × number of nodes ×
        feature size``, where the 1 is a "dummy" batch dimension and ``number of nodes`` is the real
        batch size (every node in the graph).
        """
        ...

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        """
        Create a Keras Sequence or similar input, appropriate for a graph machine learning model.
        """
        ...

    def default_corrupt_input_index_groups(self):
        """
        Optionally returns the indices of input tensors that can be shuffled for
        :class:`CorruptGenerator` to use in :class:`DeepGraphInfomax`.

        If this isn't overridden, this method returns None, indicating that the generator doesn't
        have a default or "canonical" set of indices that can be corrupted for Deep Graph Infomax.
        """
        return None
