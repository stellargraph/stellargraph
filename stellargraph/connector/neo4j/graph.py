# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

__all__ = ["Neo4jStellarGraph", "Neo4jStellarDiGraph"]

import numpy as np
import scipy.sparse as sps
import pandas as pd
from ...core.experimental import experimental
from ... import globalvar
from ...core import convert
from ...core.convert import IndexedArray
from ...core.graph import extract_element_features
from ...core.validation import comma_sep


@experimental(reason="the class is not tested", issues=[1578])
class Neo4jStellarGraph:
    """
    Neo4jStellarGraph class for graph machine learning on graphs stored in
    a Neo4j database.

    This class communicates with Neo4j via a p2neo.Graph connected to the graph
    database of interest and contains functions to query the graph data necessary
    for machine learning.

    Args:
        graph (py2neo.Graph): a py2neo.Graph connected to a Neo4j graph database.
        is_directed (bool, optional): If True, the data represents a
            directed multigraph, otherwise an undirected multigraph.
    """

    def __init__(self, graph_db, is_directed=False):

        self.graph_db = graph_db
        self._is_directed = is_directed
        self._node_feature_size = None
        self._nodes = None

        # FIXME: methods in this class currently only support homogeneous graphs with default node type
        self._node_type = globalvar.NODE_TYPE_DEFAULT

    def nodes(self):
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The node IDs of all the nodes in the graph.
        """
        node_ids_query = f"""
            MATCH (n)
            RETURN n.ID as node_id
            """

        result = self.graph_db.run(node_ids_query)
        return [row["node_id"] for row in result.data()]

    def cache_all_nodes_in_memory(self, dtype="float32"):
        """
        Load all node IDs and features into memory from Neo4j so that subsequent method calls that
        access node features can use the cached data instead of querying the database.

        This method should be avoided for larger graphs.

        Args:
            dtype (str, optional): Data type of features

        """
        features = self._node_features_from_db(None)
        self._nodes = convert.convert_nodes(
            features, name="nodes", default_type=self._node_type, dtype=dtype,
        )

        # cache feature size too
        self._node_feature_size = self._nodes.features_of_type(self._node_type).shape[1]

    def _node_features_from_db(self, nodes):
        return_node = f"""
            WITH {{ID: node.ID, features: node.features}} AS node_data
            RETURN node_data
            """
        if nodes is None:
            feature_query = f"""
                MATCH(node)
                {return_node}
                """
            result = self.graph_db.run(feature_query)
            rows = result.data()
            return IndexedArray(
                np.array([node["node_data"]["features"] for node in rows]),
                index=[node["node_data"]["ID"] for node in rows],
            )
        else:
            if isinstance(nodes, np.ndarray):
                valid = nodes != None
                # we need to create a list of integers to run the neo4j query with
                nodes = [node.item() if node is not None else node for node in nodes]
            elif isinstance(nodes, list):
                valid = np.array(nodes) != None
            else:
                # single node
                valid = np.array([nodes is not None])
                nodes = [nodes]

            # None's should be filled with zeros in the feature matrix
            features = np.zeros(
                (len(nodes), self.node_feature_sizes()[self._node_type])
            )

            # fill valid locs with features
            feature_query = f"""
                UNWIND $node_id_list AS node_id
                MATCH(node) WHERE node.ID = node_id
                {return_node}
                """
            result = self.graph_db.run(
                feature_query, parameters={"node_id_list": nodes}
            )
            rows = result.data()

            # this method currently doesn't handle any other invalid IDs. If there are other
            # invalid ids, we should raise an error
            if len(rows) != sum(valid):
                ids = {node["node_data"]["ID"] for node in rows}
                invalid = [
                    node for node in nodes if node is not None and node not in ids
                ]
                raise ValueError(
                    f"nodes: Found values that did not return any results from the database: {comma_sep(invalid)}"
                )

            features[valid, :] = [row["node_data"]["features"] for row in rows]

            return IndexedArray(features, index=nodes)

    def node_features(self, nodes):
        """
        Get the numeric feature vectors for the specified nodes or node type.

        Args:
            nodes (list or hashable, optional): Node ID or list of node IDs.
        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        if self._nodes is not None:
            return extract_element_features(
                self._nodes,
                self.unique_node_type,
                "nodes",
                nodes,
                type=None,
                use_ilocs=False,
            )
        else:
            return self._node_features_from_db(nodes)

    def unique_node_type(self):
        return self._node_type

    def node_feature_sizes(self):
        """
        Get the feature sizes for the node types in the graph.

        This method obtains the feature size by sampling a random node from the graph. Currently
        this class only supports a single default node type, and makes the following assumptions:

        - all nodes have features as a single list

        - all nodes' features have the same size

        Returns:
            A dictionary of node type and integer feature size.
        """
        if self._node_feature_size is None:
            # if feature size is unknown, take a random node's features
            feature_query = f"""
                MATCH(node)
                RETURN node.features as features LIMIT 1
                """
            result = self.graph_db.run(feature_query)
            self._node_feature_size = len(result.data()[0]["features"])

        return {self._node_type: self._node_feature_size}

    def to_adjacency_matrix(self, node_ids):
        """
        Obtains a SciPy sparse adjacency matrix for the subgraph containing
        the nodes specified in node_ids.

        Args:
            nodes (list): The collection of nodes
                comprising the subgraph. The adjacency matrix is
                computed for this subgraph.

        Returns:
             The weighted adjacency matrix.
        """

        # neo4j optimizes this query to be O(edges incident to nodes)
        # not O(E) as it appears
        subgraph_query = f"""
            MATCH (source)-->(target)
            WHERE source.ID IN $node_id_list AND target.ID IN $node_id_list
            RETURN collect(source.ID) AS sources, collect(target.ID) as targets
            """

        result = self.graph_db.run(
            subgraph_query, parameters={"node_id_list": node_ids}
        )

        data = result.data()[0]
        sources = np.array(data["sources"])
        targets = np.array(data["targets"])

        index = pd.Index(node_ids)

        src_idx = index.get_indexer(sources)
        tgt_idx = index.get_indexer(targets)

        weights = np.ones(len(sources), dtype=np.float32)
        shape = (len(node_ids), len(node_ids))
        adj = sps.csr_matrix((weights, (src_idx, tgt_idx)), shape=shape)

        if not self.is_directed() and len(data) > 0:
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops

            # FIXME https://github.com/scipy/scipy/issues/11949: these operations, particularly the
            # diagonal, don't work for an empty matrix (n == 0)
            backward = adj.transpose(copy=True)
            # this is setdiag(0), but faster, since it doesn't change the sparsity structure of the
            # matrix (https://github.com/scipy/scipy/issues/11600)
            (nonzero,) = backward.diagonal().nonzero()
            backward[nonzero, nonzero] = 0

            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def is_directed(self):
        return self._is_directed


# A convenience class that merely specifies that edges have direction.
class Neo4jStellarDiGraph(Neo4jStellarGraph):
    def __init__(self, graph_db):
        super().__init__(graph_db, is_directed=True)
