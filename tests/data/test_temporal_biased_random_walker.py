# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
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
import pytest
import networkx as nx
from stellargraph.data.explorer import TemporalBiasedRandomWalk
from stellargraph.core.graph import StellarGraph


def create_test_temporal_graph():
    """
    Creates a simple graph for testing the temporal random walks classes. The node ids are string or integers.

    :return: A multigraph where each node can have multiple edges to other nodes at different times. Each edge has an integer time stamp associated with it.
    """
    g = nx.MultiGraph()
    g.add_weighted_edges_from(
        [
            ("a", "b", 3),
            ("a", "b", 10),
            ("a", "b", 6),
            ("a", "c", 9),
            ("a", "c", 1),
            ("a", "c", 4),
            ("a", "c", 7),
            ("a", "f", 5),
            ("a", "f", 12),
            ("a", "m", 2),
            ("a", "m", 15),
            ("b", "a", 19),
            ("b", "m", 2),
            ("b", "m", 8),
            ("b", "g", 5),
            ("b", "g", 9),
            ("b", "g", 7),
            ("c", "p", 3),
            ("c", "p", 11),
            ("c", "p", 5),
            ("c", "p", 7),
            ("c", "q", 6),
            ("c", "q", 12),
            ("c", "q", 8),
            ("c", "m", 10),
            ("c", "f", 4),
            ("c", "f", 1),
            ("f", "b", 13),
            ("f", "c", 16),
            ("f", "c", 1),
            ("f", "d", 7),
            ("m", "f", 4),
            ("m", "f", 14),
            ("m", "f", 3),
            ("m", "g", 6),
            ("m", "g", 17),
            ("m", "g", 14),
            ("m", "p", 2),
            ("m", "p", 11),
            ("m", "p", 8),
            ("m", "d", 9),
            ("m", "d", 16),
            ("m", "d", 4),
        ]
    )
    g = StellarGraph(g)

    return g


class TestTemporalBiasedRandomWalk(object):
    def test_parameter_checking(self):

        g = create_test_temporal_graph()
        temporalrw = TemporalBiasedRandomWalk(g)

        nodes = list(g.nodes())
        n = 1
        length = 2
        seed = None

        # edge weight labels are by default called weight as is in networkx but they can be any string value specified by the user

        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes, n=n, length=length, seed=seed, edge_time_label=None
            )

        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes, n=n, length=length, seed=seed, edge_time_label="x"
            )

    def test_time_on_edges(self):

        # all positive walks
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 1, 4)]
        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        assert (
            len(
                temporalrw.run(
                    nodes=nodes,
                    n=n,
                    length=length,
                    seed=seed,
                    edge_time_label=edge_time_label,
                )
            )
            == 4
        )

        # edges with negative time
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, -2), (3, 4, 3), (4, 1, 4)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        with pytest.raises(ValueError):
            temporalrw.run(nodes=nodes, n=n, length=length, seed=seed)

        # edge with time infinity
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, np.inf), (3, 4, 3), (4, 1, 4)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )

        # edges with missing times
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, None), (3, 4, 3), (4, 1, 4)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)
        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )

        # edges with NaN
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, np.NaN), (3, 4, 3), (4, 1, 4)]
        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)
        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )

    def test_time_labels(self):
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 1, 4)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = list(g.nodes())
        n = 1
        length = 1
        seed = None
        edge_time_label = None

        temporalrw = TemporalBiasedRandomWalk(g)

        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )

        edge_time_label = "time"

        with pytest.raises(ValueError):
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )

    def test_temporal_walks(self):

        # time respecting walk: 1 -> 2 -> 4 -> 6
        g = nx.MultiGraph()
        edges = [(1, 2, 5), (2, 3, 2), (2, 4, 10), (4, 5, 3), (4, 6, 12)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = [1]
        n = 1
        length = 5
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        assert temporalrw.run(
            nodes=nodes, n=n, length=length, seed=seed, edge_time_label=edge_time_label
        )[0] == [1, 2, 4, 6]

    def test_biased_temporal_bidirectional_walks(self):

        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 5, 4), (5, 6, 6)]

        g.add_weighted_edges_from(edges)
        g = StellarGraph(g)

        nodes = [3]
        n = 100
        length = 5
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        #  forward (unidirectional) time respecting walk from 3: [3, 2] or [3,4,5,6]
        bidirectional = False

        assert temporalrw.run(
            nodes=nodes,
            n=n,
            length=length,
            seed=seed,
            edge_time_label=edge_time_label,
            bidirectional=bidirectional,
        )[0] in [[3, 2], [3, 4, 5, 6]]

        #  bidirectional time respecting walk from 3: [3, 2] or [1, 2, 3, 4, 5, 6]
        bidirectional = True

        walks = temporalrw.run(
            nodes=nodes,
            n=n,
            length=length,
            seed=seed,
            edge_time_label=edge_time_label,
            bidirectional=bidirectional,
        )
        assert sorted([list(i) for i in set(tuple(i) for i in walks)]) == [
            [1, 2, 3, 4, 5, 6],
            [3, 2],
        ]

    def test_isolated_nodes_walks(self):
        g = nx.MultiGraph()
        edges = [(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 5, 4), (5, 6, 6)]
        g.add_weighted_edges_from(edges)
        g.remove_edge(5, 6)
        g = StellarGraph(g)

        nodes = [6]
        n = 1
        length = 5
        seed = None
        edge_time_label = "weight"

        temporalrw = TemporalBiasedRandomWalk(g)

        assert (
            temporalrw.run(
                nodes=nodes,
                n=n,
                length=length,
                seed=seed,
                edge_time_label=edge_time_label,
            )
            == []
        )
