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

import threading

import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

from ..globalvar import SOURCE, TARGET, TYPE_ATTR_NAME


class KGTripleGenerator:
    def __init__(self, G, batch_size):
        self.G = G
        self.batch_size = batch_size

    def flow(self, edges, negative_samples=None, shuffle=False, seed=None):
        if isinstance(edges, pd.DataFrame):
            sources = edges[SOURCE]
            rels = edges[TYPE_ATTR_NAME]
            targets = edges[TARGET]
        elif isinstance(edges, tuple):
            sources, rels, targets = edges
        elif isinstance(edges, list):
            sources, rels, targets = zip(*edges)
        else:
            raise TypeError(
                f"edges: expected pandas.DataFrame, tuple or list; found {type(edges).__name__}"
            )

        source_ilocs = self.G._get_index_for_nodes(sources)
        rel_ilocs = self.G._edges.types.to_iloc(rels, strict=True)
        target_ilocs = self.G._get_index_for_nodes(targets)

        return KGTripleSequence(
            max_node_iloc=self.G.number_of_nodes(),
            source_ilocs=source_ilocs,
            rel_ilocs=rel_ilocs,
            target_ilocs=target_ilocs,
            batch_size=self.batch_size,
            shuffle=shuffle,
            negative_samples=negative_samples,
            seed=seed,
        )


class KGTripleSequence(Sequence):
    def __init__(
        self,
        *,
        max_node_iloc,
        source_ilocs,
        rel_ilocs,
        target_ilocs,
        batch_size,
        shuffle,
        negative_samples,
        seed,
    ):
        self.max_node_iloc = max_node_iloc

        num_edges = len(source_ilocs)
        self.indices = np.arange(num_edges, dtype=np.min_scalar_type(num_edges))

        self.source_ilocs = np.asarray(source_ilocs)
        self.rel_ilocs = np.asarray(rel_ilocs)
        self.target_ilocs = np.asarray(target_ilocs)

        self.negative_samples = negative_samples

        self.batch_size = batch_size
        self.seed = seed

        self.shuffle = shuffle

        self._global_rs = np.random.default_rng(seed)
        self._batch_samplers = []
        self._global_lock = threading.Lock()

    def _batch_sampler(self, batch_num):
        self._global_lock.acquire()
        try:
            return self._batch_samplers[batch_num]
        except IndexError:
            new_samplers = batch_num - len(self._batch_samplers) + 1
            seeds = self._global_rs.integers(2 ** 32, size=new_samplers)
            self._batch_samplers.extend(np.random.default_rng(seed) for seed in seeds)
            return self._batch_samplers[batch_num]
        finally:
            self._global_lock.release()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_num):
        start = self.batch_size * batch_num
        end = start + self.batch_size

        indices = self.indices[start:end]

        s_iloc = self.source_ilocs[indices]
        r_iloc = self.rel_ilocs[indices]
        o_iloc = self.target_ilocs[indices]
        positive_count = len(s_iloc)
        targets = None

        if self.negative_samples is not None:
            s_iloc = np.tile(s_iloc, 1 + self.negative_samples)
            r_iloc = np.tile(r_iloc, 1 + self.negative_samples)
            o_iloc = np.tile(o_iloc, 1 + self.negative_samples)

            negative_count = self.negative_samples * positive_count
            assert len(s_iloc) == positive_count + negative_count

            rng = self._batch_sampler(batch_num)

            change_source = rng.integers(2, size=negative_count) == 1
            source_changes = change_source.sum()
            new_nodes = rng.integers(self.max_node_iloc, size=negative_count)

            s_iloc[positive_count:][change_source] = new_nodes[:source_changes]
            o_iloc[positive_count:][~change_source] = new_nodes[source_changes:]

            targets = np.repeat(
                np.array([1, 0], dtype=np.float32), [positive_count, negative_count]
            )
            assert len(targets) == len(s_iloc)

        assert len(s_iloc) == len(r_iloc) == len(o_iloc)
        return [s_iloc, r_iloc, o_iloc], targets

    def on_epoch_end(self):
        if self.shuffle:
            self._global_rs.shuffle(self.indices)
