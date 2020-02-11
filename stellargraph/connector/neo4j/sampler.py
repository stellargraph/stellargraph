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

__all__ = [
    "Neo4JSampledBreadthFirstWalk",
    "Neo4JDirectedBreadthFirstNeighbors",
]


import numpy as np
import warnings
from collections import defaultdict, deque

from ...core.schema import GraphSchema
from ...core.graph import StellarGraph
from ...data.explorer import GraphWalk
from ...core.experimental import experimental


@experimental(reason="the class is not fully tested")
class Neo4JSampledBreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from Neo4J database.
    """

    def run(self, neo4j_graphdb, nodes=None, n=1, n_size=None, seed=None):
        """
        Send queries to Neo4J graph databases and collect sampled breadth-first walks starting from the root nodes.

        Args:
            neo4j_graphdb: (py2neo.Graph) the Neo4J Graph Database object
            nodes (list): A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n_size (int): The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            n (int, default 1): Number of walks per node id.
            neighbours with replacement is always used regardless of the node degree and number of neighbours
            requested.
            seed (int, optional): Random number generator seed; default is None

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a BFW.
        """

        def bfs_neighbor_query(sampling_direction):
            direction = {"BOTH": "--", "IN": "<--", "OUT": "-->"}

            return f"""
                // expand the list of node id in seperate rows of ids.
                UNWIND {{node_id_list}} AS node_id

                // for each node id in every row, collect the random list of its neighbors.
                CALL apoc.cypher.run(

                    'MATCH(cur_node) WHERE id(cur_node) = {{node_id}}

                    // find the neighbors
                    MATCH (cur_node){direction[sampling_direction]}(neighbors)

                    // put all ids into a list
                    WITH CASE collect(id(neighbors)) WHEN [] THEN [null] ELSE collect(id(neighbors)) END AS in_neighbors_list

                    // pick random nodes with replacement
                    WITH apoc.coll.randomItems(in_neighbors_list, {{num_samples}}, True) AS in_samples_list

                    RETURN in_samples_list',
                    {{ node_id: node_id, num_samples: {{num_samples}}  }}) YIELD value

                RETURN apoc.coll.flatten(collect(value.in_samples_list)) as next_samples
            """

        walks = [[head_node for head_node in nodes for _ in range(n)]]
        sample_query = bfs_neighbor_query(sampling_direction="BOTH")

        for num_sample in n_size:

            cur_nodes = walks[-1]

            result = neo4j_graphdb.run(
                sample_query,
                parameters={"node_id_list": cur_nodes, "num_samples": num_sample},
            )

            walks.append(result.data()[0]["next_samples"])

        return walks


@experimental(reason="the class is not fully tested")
class Neo4JDirectedBreadthFirstNeighbors(GraphWalk):
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from Neo4J database.
    """

    def run(
        self, neo4j_graphdb, nodes=None, n=1, in_size=None, out_size=None, seed=None
    ):
        """
        Send queries to Neo4J databases and collect sampled breadth-first walks starting from the root nodes.
        
        Args:
            neo4j_graphdb (py2neo.Graph): the Neo4J Graph Database object
            nodes (list): A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: (int) Number of walks per node id.
            in_size: (list) The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_size: (list) The number of out-directed nodes to sample with replacement at each depth of the walk.
            seed: (int) Random number generator seed; default is None
        Returns:
            A list of multi-hop neighbourhood samples. Each sample expresses multiple undirected walks, but the in-node
            neighbours and out-node neighbours are sampled separately. Each sample has the format:
        """

        def bfs_neighbor_query(sampling_direction):
            direction = {"BOTH": "--", "IN": "<--", "OUT": "-->"}

            return f"""
                // expand the list of node id in seperate rows of ids.
                UNWIND {{node_id_list}} AS node_id

                // for each node id in every row, collect the random list of its neighbors.
                CALL apoc.cypher.run(

                    'MATCH(cur_node) WHERE id(cur_node) = {{node_id}}

                    // find the neighbors
                    MATCH (cur_node){direction[sampling_direction]}(neighbors)

                    // put all ids into a list
                    WITH CASE collect(id(neighbors)) WHEN [] THEN [null] ELSE collect(id(neighbors)) END AS in_neighbors_list

                    // pick random nodes with replacement
                    WITH apoc.coll.randomItems(in_neighbors_list, {{num_samples}}, True) AS in_samples_list

                    RETURN in_samples_list',
                    {{ node_id: node_id, num_samples: {{num_samples}}  }}) YIELD value

                RETURN apoc.coll.flatten(collect(value.in_samples_list)) as next_samples
            """

        self._check_neighbourhood_sizes(in_size, out_size)
        self._check_common_parameters(nodes, n, len(in_size), seed)

        walks = [[head_node for head_node in nodes for _ in range(n)]]

        cur_tree_node = 0
        tree_level_end = 0

        in_sample_query = bfs_neighbor_query(sampling_direction="IN")
        out_sample_query = bfs_neighbor_query(sampling_direction="OUT")
        #
        for in_num, out_num in zip(in_size, out_size):
            while cur_tree_node <= tree_level_end:
                cur_nodes = walks[cur_tree_node]

                neighbor_records = neo4j_graphdb.run(
                    in_sample_query,
                    parameters={"node_id_list": cur_nodes, "num_samples": in_num},
                )
                walks.append(neighbor_records.data()[0]["next_samples"])

                neighbor_records = neo4j_graphdb.run(
                    out_sample_query,
                    parameters={"node_id_list": cur_nodes, "num_samples": out_num},
                )
                walks.append(neighbor_records.data()[0]["next_samples"])

                cur_tree_node += 1

            tree_level_end = tree_level_end * 2 + 2

        return walks

    def _check_neighbourhood_sizes(self, in_size, out_size):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: (list) A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n_size: (list) The number of neighbouring nodes to expand at each depth of the walk.
            seed: (int) Random number generator seed; default is None
        """
        self._check_sizes(in_size)
        self._check_sizes(out_size)
        if len(in_size) != len(out_size):
            self._raise_error(
                "The number of hops for the in and out neighbourhoods must be the same."
            )
