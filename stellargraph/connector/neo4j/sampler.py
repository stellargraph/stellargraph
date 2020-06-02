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
    "Neo4jSampledBreadthFirstWalk",
    "Neo4jDirectedBreadthFirstNeighbors",
]

from ...core.experimental import experimental
from .graph import Neo4jStellarGraph, Neo4jStellarDiGraph


def _bfs_neighbor_query(sampling_direction, id_property, node_label=None):
    """
    Generate the Cypher neighbor sampling query for a batch of nodes.

    Args:
        sampling_direction (String): indicate type of neighbors needed to sample. Direction must be 'in', 'out' or 'both'.
        id_property (str): Cypher-escaped property name for node IDs.
        node_label (str, optional): Common label for all nodes in the graph, if such label exists.
            Providing this is useful if there are any indexes created on this label (e.g. on node IDs),
            as it will improve performance of queries.
    Returns:
        The cypher query that samples the neighbor ids for a batch of nodes.
    """
    direction_arrow = {"BOTH": "--", "IN": "<--", "OUT": "-->"}[sampling_direction]
    match_cur_node = (
        "MATCH(cur_node)" if node_label is None else f"MATCH(cur_node:{node_label})"
    )

    return f"""
        // expand the list of node id in seperate rows of ids.
        UNWIND $node_id_list AS node_id

        // for each node id in every row, collect the random list of its neighbors.
        CALL apoc.cypher.run(

            '{match_cur_node} WHERE cur_node.{id_property} = $node_id

            // find the neighbors
            MATCH (cur_node){direction_arrow}(neighbors)

            // put all ids into a list
            WITH CASE collect(neighbors.{id_property}) WHEN [] THEN [null] ELSE collect(neighbors.{id_property}) END AS in_neighbors_list

            // pick random nodes with replacement
            WITH apoc.coll.randomItems(in_neighbors_list, $num_samples, True) AS in_samples_list

            RETURN in_samples_list',
            {{ node_id: node_id, num_samples: $num_samples  }}) YIELD value

        RETURN apoc.coll.flatten(collect(value.in_samples_list)) as next_samples
        """


@experimental(reason="the class is not fully tested")
class Neo4jSampledBreadthFirstWalk:
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from Neo4j database.
    """

    def __init__(self, graph):
        if not isinstance(graph, Neo4jStellarGraph):
            raise TypeError("Graph must be a Neo4jStellarGraph or Neo4jStellarDiGraph.")
        self.graph = graph

    def run(self, nodes=None, n=1, n_size=None):
        """
        Send queries to Neo4j graph databases and collect sampled breadth-first walks starting from
        the root nodes.

        Args:
            nodes (list of hashable): A list of root node ids such that from each node n BFWs will
                be generated up to the given depth d.
            n_size (list of int): The number of neighbouring nodes to expand at each depth of the
                walk. Sampling of neighbours with replacement is always used regardless of the node
                degree and number of neighbours requested.
            n (int): Number of walks per node id.
            seed (int, optional): Random number generator seed; default is None

        Returns:
            A list of lists, each list is a sequence of sampled node ids at a certain hop.
        """

        samples = [[head_node for head_node in nodes for _ in range(n)]]
        neighbor_query = _bfs_neighbor_query(
            sampling_direction="BOTH",
            id_property=self.graph.cypher_id_property,
            node_label=self.graph.cypher_node_label,
        )

        # this sends O(number of hops) queries to the database, because the code is cleanest like that
        for num_sample in n_size:
            cur_nodes = samples[-1]
            result = self.graph.graph_db.run(
                neighbor_query,
                parameters={"node_id_list": cur_nodes, "num_samples": num_sample},
            )
            samples.append(result.data()[0]["next_samples"])

        return samples


@experimental(reason="the class is not fully tested")
class Neo4jDirectedBreadthFirstNeighbors:
    """
    Breadth First Walk that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes from Neo4j database.
    """

    def __init__(self, graph):
        if not isinstance(graph, Neo4jStellarDiGraph):
            raise TypeError("Graph must be a Neo4jStellarDiGraph.")
        self.graph = graph

    def run(self, nodes=None, n=1, in_size=None, out_size=None):
        """
        Send queries to Neo4j databases and collect sampled breadth-first walks starting from the root nodes.

        Args:
            nodes (list of hashable): A list of root node ids such that from each node n BFWs will
                be generated up to the given depth d.
            n (int): Number of walks per node id.
            in_size (list of int): The number of in-directed nodes to sample with replacement at each depth of the walk.
            out_size (list of int): The number of out-directed nodes to sample with replacement at each depth of the walk.
        Returns:
            A list of multi-hop neighbourhood samples. Each sample expresses a collection of nodes, which could be either in-neighbors,
            or out-neighbors of the previous hops.
            Result has the format:
            [[head1, head2, ...],
            [in1_head1, in2_head1, ..., in1_head2, in2_head2, ...], [out1_head1, out2_head1, ..., out1_head2, out2_head2, ...],
            [in1_in1_head1, in2_in1_head1, ..., in1_in2_head1, ...], [out1_in1_head1, out2_in1_head1, ..., out1_in2_head1, ...],
            [in1_out1_head1, in2_out1_head1, ..., in1_out2_head1, ...], [out1_out1_head1, out2_out1_head1, ..., out1_out2_head1, ...],
            ...
            ]
        """
        # FIXME: we may want to run validation on all the run parameters similar to other GraphWalk classes

        head_nodes = [head_node for head_node in nodes for _ in range(n)]
        hops = [[head_nodes]]

        in_sample_query = _bfs_neighbor_query(
            sampling_direction="IN",
            id_property=self.graph.cypher_id_property,
            node_label=self.graph.cypher_node_label,
        )
        out_sample_query = _bfs_neighbor_query(
            sampling_direction="OUT",
            id_property=self.graph.cypher_id_property,
            node_label=self.graph.cypher_node_label,
        )

        # this sends O(2^number of hops) queries to the database, because the code is cleanest like that
        for in_num, out_num in zip(in_size, out_size):
            last_hop = hops[-1]
            this_hop = []
            for cur_nodes in last_hop:
                # get in-neighbor nodes
                neighbor_records = self.graph.graph_db.run(
                    in_sample_query,
                    parameters={"node_id_list": cur_nodes, "num_samples": in_num},
                )
                this_hop.append(neighbor_records.data()[0]["next_samples"])

                # get out-neighbor nodes
                neighbor_records = self.graph.graph_db.run(
                    out_sample_query,
                    parameters={"node_id_list": cur_nodes, "num_samples": out_num},
                )
                this_hop.append(neighbor_records.data()[0]["next_samples"])

            hops.append(this_hop)

        return sum(hops, [])
