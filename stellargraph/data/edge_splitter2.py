# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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

__all__ = ["EdgeSplitter"]

import datetime
import warnings
import networkx as nx
import pandas as pd
import numpy as np
from math import isclose

from ..core.graph import StellarGraph
from ..core.element_data import EdgeData


class EdgeSplitter:
    def __init__(self, g, g_master=None):
        self.g = g
        self.g_master = g_master

    @property
    def reference_graph(self):
        if self.g_master is None:
            return self.g

        return self.g_master

    def train_test_split(
        p=0.5,
        method="global",
        probs=None,
        keep_connected=False,
        edge_label=None,
        edge_attribute_label=None,
        edge_attribute_threshold=None,
        attribute_is_datetime=None,
        seed=None,
    ):
        if not (0 < p < 1):
            raise ValueError(f"p: expected value in the interval (0, 1), found {p!r}")

        if method not in ("global", "local"):
            raise ValueError(f"method: expected 'global' or 'local', found {method!r}")

        if not isinstance(keep_connected, bool):
            raise TypeError(
                f"keep_connected: expected bool, found {type(keep_connected).__name__}"
            )

        try:
            random = np.random.RandomState(seed)
        except Exception:
            raise ValueError(f"seed: expected integer in [0, 2**32 - 1), found {seed}")

        if keep_connected:
            raise NotImplementedError()

        if edge_label is not None:
            return self._train_test_split_heterogeneous(
                p, method, edge_label, edge_attribute_threshold, random
            )

        return self._train_test_split_homogeneous(
            p, method, probs, edge_attribute_threshold, random
        )

    def _train_test_split_heterogeneous(
        self, p, method, edge_label, edge_attribute_threshold, random
    ):
        raise NotImplementedError()

    def _negative_global(self, num_test_edges, random):
        edges = self.reference_graph.edges()

        nonnegative_edges = set(edges)
        nonnegative_edges.update((t, s) for s, t in edges)
        # self-loops don't count as negative edges either
        nonnegative_edges.update((n, n) for n in self.reference_graph.nodes())

        negative_test_edges = set()
        source = np.asarray(self.reference_graph.nodes())
        target = source.copy()

        num_iter = int(np.ceil(num_test_edges) / len(source)) + 1
        for _ in range(num_iter):
            random.shuffle(source)
            random.shuffle(target)
            for u, v in zip(source, target):
                if (u, v) not in nonnegative_edges:
                    negative_test_edges.add((u, v))

                if len(negative_test_edges) == num_test_edges:
                    return list(negative_test_edges)

        raise ValueError(
            f"failed to sample {num_test_edges} negative edges. Consider using a smaller value for p."
        )

    def _train_test_split_homogeneous(
        self, p, method, probs, edge_attribute_threshold, random
    ):
        num_edges = self.g.number_of_edges()

        num_test_edges = int(num_edges * p)

        random_edge_ilocs = random.permutation(num_edges)

        positive_test_ilocs = random_edge_ilocs[:num_test_edges]
        remaining_ilocs = random_edge_ilocs[num_test_edges:]

        if method == "global":
            negative_test_edges = _negative_global(num_test_edges, random)
        else:
            raise NotImplementedError()

        node_type = next(iter(self.g.node_types))
        raw_edges = self.reference_graph._edges
        remaining_edge_data = EdgeData(
            {
                node_type: pd.DataFrame(
                    {
                        SOURCE: raw_edges.sources[remaining_ilocs],
                        TARGET: raw_edges.targets[remaining_ilocs],
                        WEIGHT: raw_edges.weights[remaining_ilocs],
                    }
                )
            }
        )

        positive_test_edges = list(
            zip(
                raw_edges.sources[positive_test_ilocs],
                raw_edges.targets[positive_test_ilocs],
            )
        )
        test_edges = positive_test_edges + negative_test_edges
        assert len(positive_test_edges) == len(negative_test_edges) == num_test_edges
        test_labels = np.array([1, 0]).repeat(num_test_edges)

        return (
            type(self.g)(
                _internal_raw_data=(self.reference_graph._nodes, remaining_edge_data)
            ),
            test_edges,
            test_labels,
        )
