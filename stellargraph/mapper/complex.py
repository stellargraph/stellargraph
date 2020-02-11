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

class ComplExSequence(Sequence):
    def __init__(self, G, batch_size, negative=None, seed=None):
        self.G = G

        source_ilocs = G._get_index_for_nodes(G._edges.sources)
        rel_ilocs = G._edges.type_ilocs
        target_ilocs = G._get_index_for_nodes(G._edges.targets)
        self.triples = np.column_stack([source_ilocs, rel_ilocs, target_ilocs])

        if negative is None:
            self.negatives = 0
        else:
            self.negatives = negative

        self.batch_size = batch_size
        self.seed = seed

        self._batch_sampler_rs = np.random.default_rng(seed)
        self._batch_samplers = []

    def _batch_sampler(self, batch_num):
        try:
            return self._batch_sampler[batch_num]
        except IndexError:
            for _ in range(len(self._batch_samplers), batch_num):
                self._batch_samplers.append(np.random.default_rng(self._batch_sampler_rs.integers(2**32)))

            return self._batch_sampler[batch_num]

    def __len__(self):
        return np.ceil(len(self.edges) / self.batch_size)

    def __getitem__(self, batch_num):
        start = self.batch_size * batch_num
        end = start + batch_num

        values = self.triples[start:end, :]
        positive_count = len(values)
        targets = None

        if self.negatives > 0:
            values = np.tile(values, (1 + self.negatives))
            negative_count = self.negatives * positive_count
            assert len(values) == positive_count + negative_count

            rng = self._batch_sampler(batch_num)

            change_source = rng.integers(2, size=negative_count) == 1
            source_changes = change_source.sum()
            new_nodes = rng.integers(G.number_of_nodes(), size=negative_count)

            values[positive_count:, 0][change_source] = new_nodes[:source_changes]
            values[positive_count:, 2][~change_source] = new_nodes[source_changes:]

            targets = np.repeat([1, -1], [positive_count, negative_count])

        return values



