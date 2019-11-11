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
__all__ = ["StellarGraph", "StellarDiGraph"]

from typing import Iterable, Any, Mapping, Optional, Union

from .schema import GraphSchema


class StellarGraphFactory(type):
    """
    Private class for instantiating the StellarGraph interface from
    user-supplied information.
    """

    def __call__(cls, *args, **kwargs):
        if cls is StellarGraph or cls is StellarDiGraph:
            is_directed = cls is StellarDiGraph
            if StellarGraphFactory.is_networkx(args, kwargs):
                # XXX Import is here to avoid circular definitions
                from .graph_networkx import NetworkXStellarGraph

                return NetworkXStellarGraph(*args, is_directed=is_directed, **kwargs)
            else:
                from .graph_standard import StandardStellarGraph

                return StandardStellarGraph(is_directed, *args, **kwargs)
        else:
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def is_networkx(args, kwargs):
        # TODO Actually check for a NetworkX graph instance
        # For now, test for legacy interface:
        return len(args) == 1 or "graph" in kwargs


class StellarGraph(metaclass=StellarGraphFactory):
    # TODO Update docs for new interface
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
        graph: The positional or keyword argument specifying the NetworkX graph instance.
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

    def nodes(
        self, include_info: bool = False
    ) -> Union[Iterable[Any], Iterable[tuple]]:
        """
        Obtains the collection of nodes in the graph.

        Args:
            include_info (bool): If False (default), then only the node identifiers
            are provided; otherwise, both node identifiers and node types are
            provided.
        Returns:
             The collection of node information.
        """
        raise NotImplementedError

    def edges(self, include_info: bool = False) -> Iterable[tuple]:
        """
        Obtains the edges in the graph, where each edge
        is represented by a tuple containing (at least)
        the source and target node identifiers.

        Args:
            include_info (bool):
                If True, then the edge information also contains
                the edge identifier, type and weight.
        Returns:
             The collection of edges.
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

    def node_type(self, node: Any) -> Any:
        """
        Obtains the type of the given node.

        Args:
            node (any): The node identifier.

        Returns:
            any: The node type.
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

    def info(self, show_attributes: bool = True, sample: Optional[int] = None):
        """
        Obtains an information string summarising information on the current graph.
        This includes node and edge type information and optionally attributes.

        Note: This potentially requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes (bool, optional):
                Indicates whether or not to display node/edge attributes (defaults to True).
            sample (int, optional):
                To speed up the graph analysis, use only a random sample of
                this many nodes and edges.

        Returns:
            An information string.
        """
        raise NotImplementedError

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        raise NotImplementedError

    def adjacency_weights(self):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        Returns:
             The weighted adjacency matrix.
        """
        raise NotImplementedError

    ##################################################################
    # Private methods:

    def check_graph_for_ml(self):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        raise NotImplementedError

    def create_graph_schema(
        self, create_type_maps: bool = True, nodes: Optional[Iterable[Any]] = None
    ) -> GraphSchema:
        """
        Creates a graph schema from current graph.

        Note: the assumption we make that there is only one
        edge of a particular edge type per node pair.

        This means that specifying an edge by source, target and edge type
        is unique.

        Arguments:
            create_type_maps (bool): If True, a lookup of node/edge types is
                created in the schema. This can be slow.

            nodes (iterable): A collection of node identifiers to use to build schema.
                This must represent all node types and all edge types in the graph.
                If specified, `create_type_maps` must be False.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema: The graph schema.
        """
        raise NotImplementedError

    def node_feature_sizes(self) -> Mapping[Any, int]:
        """
        Obtains a mapping from node types to node feature sizes.

        Returns:
             The node-type -> number-of-features mapping; this is
             empty if node features are unavailable.
        """
        raise NotImplementedError

    def node_features(self, nodes: Iterable[Any], node_type: Any = None):
        """
        Obtains the numeric feature vectors for the specified nodes as a NumPy array.

        Args:
            nodes (iterable): A collection of node identifiers.
            node_type (any, optional): The common type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        raise NotImplementedError


class StellarDiGraph(StellarGraph):
    pass
