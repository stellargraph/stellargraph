# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Data61, CSIRO
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


class StellarGraphFactory(type):
    """
    Private class for instantiating the StellarGraph interface from
    user-supplied information.
    """

    def __call__(cls, *args, **kwargs):
        if cls is StellarGraph or cls is StellarDiGraph:
            if "is_directed" in kwargs:
                raise ValueError("Restricted keyword 'is_directed'")
            is_directed = cls is StellarDiGraph
            # XXX Import is here to avoid circular definitions
            from .graph_networkx import NetworkXStellarGraph

            return NetworkXStellarGraph(*args, is_directed=is_directed, **kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)


class StellarGraph(metaclass=StellarGraphFactory):
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

    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        raise NotImplementedError

    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        raise NotImplementedError

    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        raise NotImplementedError

    def nodes(self) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The graph nodes.
        """
        raise NotImplementedError

    def edges(self) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Returns:
            The graph edges.
        """
        raise NotImplementedError

    def has_node(self, node: Any) -> bool:
        """
        Indicates whether or not the graph contains the specified node.

        Args:
            node (any): The node.

        Returns:
             bool: A value of True (cf False) if the node is
             (cf is not) in the graph.
        """
        raise NotImplementedError

    def neighbour_nodes(self, node: Any) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.

        Returns:
            iterable: The neighbouring nodes.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    ##################################################################
    # Computationally intensive methods:

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        raise NotImplementedError

    def adjacency_weights(self, nodes: Optional[Iterable] = None):
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
        raise NotImplementedError


# A convenience class that merely specifies that edges have direction.
class StellarDiGraph(StellarGraph):
    pass
