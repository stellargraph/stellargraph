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

"""
The StellarGraph class that encapsulates information required for
a machine-learning ready graph used by models.

"""
__all__ = ["StellarGraph", "StellarDiGraph", "GraphSchema"]

from typing import Iterable, Any, Mapping, Optional

from .schema import GraphSchema


class StellarGraph:
    """
    StellarGraph class for directed or undirected graph ML models. It stores both
    graph structure and features for machine learning.

    To create a StellarGraph object ready for machine learning, at a
    minimum pass the graph structure to the StellarGraph as a NetworkX
    graph:

    For undirected models::

        Gs = StellarGraph(nx_graph)


    For directed models::

        Gs = StellarDiGraph(nx_graph)


    To create a StellarGraph object with node features, supply the features
    as a numeric feature vector for each node.

    To take the feature vectors from a node attribute in the original NetworkX
    graph, supply the attribute name to the ``node_features`` argument::

        Gs = StellarGraph(nx_graph, node_features="feature")


    where the nx_graph contains nodes that have a "feature" attribute containing
    the feature vector for the node. All nodes of the same type must have
    the same size feature vectors.

    Alternatively, supply the node features as Pandas DataFrame objects with
    the of the DataFrame set to the node IDs. For graphs with a single node
    type, you can supply the DataFrame object directly to StellarGraph::

        node_data = pd.DataFrame(
            [feature_vector_1, feature_vector_2, ..],
            index=[node_id_1, node_id_2, ...])
        Gs = StellarGraph(nx_graph, node_features=node_data)

    For graphs with multiple node types, provide the node features as Pandas
    DataFrames for each type separately, as a dictionary by node type.
    This allows node features to have different sizes for each node type::

        node_data = {
            node_type_1: pd.DataFrame(...),
            node_type_2: pd.DataFrame(...),
        }
        Gs = StellarGraph(nx_graph, node_features=node_data)


    You can also supply the node feature vectors as an iterator of `node_id`
    and feature vector pairs, for graphs with single and multiple node types::

        node_data = zip([node_id_1, node_id_2, ...],
            [feature_vector_1, feature_vector_2, ..])
        Gs = StellarGraph(nx_graph, node_features=node_data)


    Args:
        graph: The NetworkX graph instance.
        node_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the node types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the nodes of the graph to determine
            their type.

        node_type_default: str, optional (default=globals.NODE_TYPE_DEFAULT)
            This is the default node type to use for nodes that do not have
            an explicit type.

        edge_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the edge types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the edges of the graph to determine
            their type.

        edge_type_default: str, optional (default=globals.EDGE_TYPE_DEFAULT)
            This is the default edge type to use for edges that do not have
            an explicit type.

        node_features: str, dict, list or DataFrame optional (default=None)
            This tells StellarGraph where to find the node feature information
            required by some graph models. These are expected to be
            a numeric feature vector for each node in the graph.

    """
    def __init__(self, *args, **kwargs):
        from .graph_networkx import NetworkXStellarGraph
        is_directed = kwargs.pop("is_directed", False)
        self._graph = NetworkXStellarGraph(is_directed=is_directed, *args, **kwargs)

    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        return self._graph.is_directed()

    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        return self._graph.number_of_edges()

    def nodes(self) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The graph nodes.
        """
        return self._graph.nodes()

    def edges(self, triple=False) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Args:
            triple (bool): A flag that indicates whether to return edge triples
            of format (node 1, node 2, edge type) or edge pairs of format (node 1, node 2).

        Returns:
            The graph edges.
        """
        return self._graph.edges(triple)

    def has_node(self, node: Any) -> bool:
        """
        Indicates whether or not the graph contains the specified node.

        Args:
            node (any): The node.

        Returns:
             bool: A value of True (cf False) if the node is
             (cf is not) in the graph.
        """
        return self._graph.has_node(node)

    def neighbors(self, node: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.

        Returns:
            iterable: The neighbouring nodes.
        """
        return self._graph.neighbors(node)

    def in_nodes(self, node: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.

        Returns:
            iterable: The neighbouring in-nodes.
        """
        return self._graph.in_nodes(node)

    def out_nodes(self, node: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.

        Returns:
            iterable: The neighbouring out-nodes.
        """
        return self._graph.out_nodes(node)

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type (hashable, optional): a type of nodes that exist in the graph

        Returns:
            A list of node IDs with type node_type
        """
        return self._graph.nodes_of_type(node_type)

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        return self._graph.node_type(node)

    @property
    def node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
        return self._graph.node_types()

    def node_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types (list, optional): A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        return self._graph.node_feature_sizes(node_types)

    def node_features(self, nodes, node_type=None):
        """
        Get the numeric feature vectors for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            nodes (list or hashable): Node ID or list of node IDs
            node_type (hashable): the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        return self._graph.node_features(nodes, node_types)

    ##################################################################
    # Computationally intensive methods:

    def info(self, show_attributes=True, sample=None):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes (bool, default True): If True, include attributes information
            sample (int): To speed up the graph analysis, use only a random sample of
                          this many nodes and edges.

        Returns:
            An information string.
        """
        return self._graph.info(show_attributes, sample)

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        return self._graph.node_degrees()

    def to_adjacency_matrix(self, nodes: Optional[Iterable] = None):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        Args:
            nodes (iterable): The optional collection of nodes
                comprising the subgraph. If specified, then the
                adjacency matrix is computed for the subgraph;
                otherwise, it is computed for the full graph.

        Returns:
             The weighted adjacency matrix.
        """
        return self._graph.to_adjacency_matrix(nodes)


# A convenience class that merely specifies that edges have direction.
class StellarDiGraph(StellarGraph):
    def __init__(self, *args, **kwargs):
        kwargs["is_directed"] = True
        super().__init__(*args, **kwargs)
