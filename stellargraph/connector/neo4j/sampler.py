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
import pandas as pd
import random
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

    def breadth_first_search_query_builder(self, num_samples):
        """
        Build the Cypher query to perform undirected breadth first search from a list of root nodes.

        Args:
            num_samples: An iterable of head nodes to perform sampling on.

        Returns:
            Cypher query to retrieve all node ids of the sampled subgraph.
        """

        query = "\n".join(
            (
                # find the head_nodes in the database
                "MATCH(head_node) WHERE ID(head_node) IN {head_nodes}",
                # place all nodes into a list
                "WITH apoc.coll.flatten(collect(head_node)) AS cur_hop_node_list,",
                # add the list of node ids into subgraph
                "[apoc.coll.flatten(collect(id(head_node)))] AS subgraph",
            )
        )

        for n in num_samples:
            next_hop_query = "\n".join(
                (
                    "UNWIND (CASE cur_hop_node_list WHEN [] THEN [null] ELSE cur_hop_node_list END) AS cur_node",
                    "CALL apoc.cypher.run(",
                    "     'WITH {cur_node} AS cur_node MATCH (cur_node)-[]-(neighbors)",
                    f"      WITH apoc.coll.randomItems(collect(neighbors), {n}, True) AS neighbor_list",
                    "      WITH (CASE neighbor_list WHEN [] THEN [null] ELSE neighbor_list END) AS neighbor_list",
                    "      UNWIND neighbor_list AS neighbor_node",
                    "      RETURN neighbor_list AS neighbor_list,",
                    # put all ids of the neighbors into a list, or create a list of n null elements if no elements found
                    f"     (CASE collect(id(neighbor_node)) WHEN [] THEN [ _ in range(0, {n} - 1) | null]",
                    "      ELSE collect(id(neighbor_node)) END) AS neighbor_id_list',",
                    "{cur_node: cur_node}) YIELD value",
                    "WITH apoc.coll.flatten(collect(value.neighbor_list)) AS cur_hop_node_list,",
                    "subgraph + [apoc.coll.flatten(collect(value.neighbor_id_list))] AS subgraph",
                )
            )
            query = "\n".join((query, next_hop_query))

        return "\n".join((query, "return subgraph"))

    def run(self, neo4j_graphdb, nodes=None, n=1, n_size=None, seed=None):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

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
        self._check_sizes(n_size)
        self._check_common_parameters(nodes, n, len(n_size), seed)

        bfsQuery = self.breadth_first_search_query_builder(n_size)

        result_records = neo4j_graphdb.run(bfsQuery, {"head_nodes": nodes})
        walks = pd.DataFrame(result_records)[0][0]

        return walks


@experimental(reason="the class is not fully tested")
class Neo4JDirectedBreadthFirstNeighbors(GraphWalk):
    """
    Breadth First sampler that generates the composite of a number of sampled paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from the Neo4J graph database.
    """

    def __init__(self, graph, graph_schema=None, seed=None):
        super().__init__(graph, graph_schema, seed)
        if not graph.is_directed():
            self._raise_error("Graph must be directed")

    def directed_breadth_first_search_query_builder(self, in_samples, out_samples):
        """
        Build the Cypher query to perform a sampled directed breadth-first search walk from root nodes.

        Args:
            in_samples (int): The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_samples (int): The number of out-directed nodes to sample with replacement at each depth of the walk.

        Returns:
            Cypher query to retrieve all node ids of the sampled subgraph.
        """

        query = "\n".join(
            (
                # find the head_nodes
                "OPTIONAL MATCH (head_node) WHERE ID(head_node) in {head_nodes}",
                # put the current level of nodes into a list of list
                "WITH [apoc.coll.flatten(collect(head_node))] AS current_depth_list,",
                # add the list of node ids into subgraph
                "[apoc.coll.flatten(collect(id(head_node)))] AS subgraph",
            )
        )

        for in_node, out_node in zip(in_samples, out_samples):
            next_hop_query = "\n".join(
                (
                    "UNWIND current_depth_list AS cur_node_list",
                    'CALL apoc.cypher.run("',
                    "WITH {cur_node_list} AS cur_node_list",
                    "UNWIND cur_node_list AS current_depth_child",
                    # call the subprocedure to collect random in-nodes.
                    "CALL apoc.cypher.run(",
                    "    'WITH {current_node} AS current_node",
                    # find all in-neighbors
                    "     MATCH (current_node)<-[]-(in_neighbors)",
                    # turn the rows into a list
                    "     WITH CASE collect(in_neighbors) WHEN [] THEN [null] ELSE collect(in_neighbors) END AS in_neighbors_list",
                    # choose random nodes from the list with replacement
                    "     WITH apoc.coll.randomItems(in_neighbors_list, {in_nodes}, True) AS in_samples_list",
                    # turn the list into rows for collecting ids
                    "     UNWIND in_samples_list AS in_samples",
                    "     RETURN in_samples_list,",
                    "     (CASE collect(id(in_samples)) WHEN [] THEN [ _ in range(0, {in_nodes} - 1) | null]",
                    "     ELSE collect(id(in_samples)) END) AS in_samples_id_list',",
                    "    { current_node: current_depth_child, in_nodes: {in_nodes} }",
                    "    ) YIELD value",
                    "RETURN apoc.coll.flatten(collect(value.in_samples_list)) AS samples_list,",
                    "    apoc.coll.flatten(collect(value.in_samples_id_list)) AS samples_id_list",
                    "UNION ALL",
                    "WITH {cur_node_list} AS cur_node_list",
                    "UNWIND cur_node_list AS current_depth_child",
                    # call the subprocedure to collect random out-nodes.
                    "CALL apoc.cypher.run(",
                    "    'WITH {current_node} AS current_node",
                    # find all out-neighbors
                    "     MATCH (current_node)-[]->(out_neighbors)",
                    # turn the rows into a list
                    "     WITH CASE collect(out_neighbors) WHEN [] THEN [null] ELSE collect(out_neighbors) END AS out_neighbors_list",
                    # choose random nodes from the list with replacement
                    "     WITH apoc.coll.randomItems((CASE out_neighbors_list WHEN [] THEN [null] ELSE out_neighbors_list END), {out_nodes}, True) AS out_samples_list",
                    # turn the list into rows for collecting ids
                    "     UNWIND out_samples_list AS out_samples",
                    "     RETURN out_samples_list,",
                    "     (CASE collect(id(out_samples)) WHEN [] THEN [ _ in range(0, {out_nodes} - 1) | null]",
                    "     ELSE collect(id(out_samples)) END) AS out_samples_id_list',",
                    "    {current_node: current_depth_child, out_nodes: {out_nodes} }",
                    "    ) YIELD value",
                    "RETURN apoc.coll.flatten(collect(value.out_samples_list)) AS samples_list,",
                    '       apoc.coll.flatten(collect(value.out_samples_id_list)) AS samples_id_list",',
                    f" {{cur_node_list: cur_node_list, in_nodes: {in_node}, out_nodes: {out_node} }}) YIELD value",
                    "WITH apoc.coll.flatten(collect([value.samples_list])) AS current_depth_list, subgraph + apoc.coll.flatten(collect([value.samples_id_list])) AS subgraph",
                )
            )

            query = "\n".join((query, next_hop_query))

        return "\n".join((query, "return subgraph"))

    def run(
        self, neo4j_graphdb, nodes=None, n=1, in_size=None, out_size=None, seed=None
    ):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

        Args:
            neo4j_graphdb: (py2neo.Graph) the Neo4J Graph Database object
            nodes:  (list) A list of root node ids such that from each node n BFWs will be generated up to the
            given depth d.
            n: (int) Number of walks per node id.
            in_size: (list) The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_size: (list) The number of out-directed nodes to sample with replacement at each depth of the walk.
            seed: (int) Random number generator seed; default is None


        Returns:
            A list of multi-hop neighbourhood samples. Each sample expresses multiple undirected walks, but the in-node
            neighbours and out-node neighbours are sampled separately. Each sample has the format:

                [[node]
                 [in_1...in_n]  [out_1...out_m]
                 [in_1.in_1...in_n.in_p] [in_1.out_1...in_n.out_q]
                    [out_1.in_1...out_m.in_p] [out_1.out_1...out_m.out_q]
                 [in_1.in_1.in_1...in_n.in_p.in_r] [in_1.in_1.out_1...in_n.in_p.out_s] ...
                 ...]

            where a single, undirected walk might be, for example:

                [node out_i  out_i.in_j  out_i.in_j.in_k ...]
        """
        self._check_neighbourhood_sizes(in_size, out_size)
        self._check_common_parameters(nodes, n, len(in_size), seed)

        bfsQuery = self.directed_breadth_first_search_query_builder(in_size, out_size)

        result_records = neo4j_graphdb.run(bfsQuery, {"head_nodes": nodes})
        samples = pd.DataFrame(result_records)[0][0]

        return [samples]

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
