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
__all__ = ["StellarGraph", "StellarDiGraph", "GraphSchema", "NeighbourWithWeight"]

from typing import Iterable, Any, Mapping, List, Optional, Set
from collections import namedtuple
import pandas as pd
import numpy as np
import scipy.sparse as sps

from .. import globalvar
from .schema import GraphSchema, EdgeType
from .experimental import experimental
from .element_data import NodeData, EdgeData, ExternalIdIndex
from .utils import is_real_iterable
from .validation import comma_sep


NeighbourWithWeight = namedtuple("NeighbourWithWeight", ["node", "weight"])

_WeightConfig = namedtuple("_WeightConfig", ["default", "label"])


def _convert_single(type_name, data, *, shared_cols, weight_config, parent_name):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{parent_name}[{type_name!r}]: expected pandas DataFrame, found {type(data)}"
        )

    if weight_config is not None:
        if weight_config.label not in data.columns:
            data = data.assign(**{globalvar.WEIGHT: weight_config.default})
        else:
            data = data.rename(columns={weight_config.label: globalvar.WEIGHT})

    shared = data[shared_cols]
    features = data.drop(columns=shared_cols).to_numpy()
    return shared, features


def _convert_elements(elements, *, default_type, shared_cols, weight_config, name):
    if isinstance(elements, pd.DataFrame):
        elements = {default_type: elements}

    if not isinstance(elements, dict):
        raise TypeError(f"{name}: expected dict, found {type(elements)}")

    singles = {
        type_name: _convert_single(
            type_name,
            data,
            weight_config=weight_config,
            shared_cols=shared_cols,
            parent_name=name,
        )
        for type_name, data in elements.items()
    }
    return (
        {type_name: shared for type_name, (shared, _) in singles.items()},
        {type_name: features for type_name, (_, features) in singles.items()},
    )


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

    def __init__(
        self,
        graph=None,
        is_directed=False,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
        nodes=None,
        edges=None,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
    ):
        if graph is not None or (nodes is None and edges is None):
            # While migrating, maintain complete compatibility by deferring to the networkx
            # implementation

            # Avoid a circular import
            from .graph_networkx import NetworkXStellarGraph

            self._graph = NetworkXStellarGraph(
                graph,
                is_directed,
                edge_weight_label,
                node_type_name,
                edge_type_name,
                node_type_default,
                edge_type_default,
                feature_name,
                target_name,
                node_features,
                dtype,
            )
        else:
            self._graph = None

            if nodes is None:
                nodes = {}
            if edges is None:
                edges = {}

            self._is_directed = is_directed

            node_shared, node_features = _convert_elements(
                nodes,
                default_type=node_type_default,
                shared_cols=[],
                weight_config=None,
                name="nodes",
            )
            self._nodes = NodeData(node_shared, node_features)
            edge_shared, edge_features = _convert_elements(
                edges,
                default_type=edge_type_default,
                shared_cols=[source_column, target_column, edge_weight_label],
                weight_config=_WeightConfig(1.0, edge_weight_label),
                name="edges",
            )
            for key, value in edge_features.items():
                _rows, columns = value.shape
                if columns:
                    raise ValueError(
                        f"edges[{key!r}]: expected zero feature columns, found {columns}"
                    )

            self._edges = EdgeData(edge_shared, self._nodes)

    # customise how a missing attribute is handled to give better error messages for the NetworkX
    # -> no NetworkX transition.
    def __getattr__(self, item):
        import networkx

        try:
            # do the normal access, in case the attribute actually exists, and to get the native
            # python wording of the error
            return super().__getattribute__(item)
        except AttributeError as e:
            if hasattr(networkx.MultiDiGraph, item):
                # a networkx class has this as an attribute, so let's assume that it's old code
                # from before the conversion and replace (the `from None`) the default exception
                # with one with a more specific message that guides the user to the fix
                type_name = type(self).__name__
                raise AttributeError(
                    f"{e.args[0]}. The '{type_name}' type no longer inherits from NetworkX types: use a new StellarGraph method, or, if that is not possible, the `.to_networkx()` conversion function."
                ) from None

            # doesn't look like a NetworkX method so use the default error
            raise

    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        if self._graph is not None:
            return self._graph.is_directed()

        return self._is_directed

    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        if self._graph is not None:
            return self._graph.number_of_nodes()

        return len(self._nodes)

    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        if self._graph is not None:
            return self._graph.number_of_edges()

        return len(self._edges)

    def nodes(self) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Returns:
            The graph nodes.
        """
        if self._graph is not None:
            return self._graph.nodes()

        return self._nodes.ids.pandas_index

    def edges(self, triple=False) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Args:
            triple (bool): A flag that indicates whether to return edge triples
            of format (node 1, node 2, edge type) or edge pairs of format (node 1, node 2).

        Returns:
            The graph edges.
        """
        if self._graph is not None:
            return self._graph.edges(triple)

        # FIXME: these would be better returned as the 2 or 3 arrays directly, rather than tuple-ing
        # (the same applies to all other instances of zip in this file)
        if triple:
            return list(
                zip(
                    self._edges.sources,
                    self._edges.sources,
                    self._edges.type_of_iloc(slice(None)),
                )
            )

        return list(zip(self._edges.sources, self._edges.targets))

    def has_node(self, node: Any) -> bool:
        """
        Indicates whether or not the graph contains the specified node.

        Args:
            node (any): The node.

        Returns:
             bool: A value of True (cf False) if the node is
             (cf is not) in the graph.
        """
        if self._graph is not None:
            return self._graph.has_node(node)

        return node in self._nodes

    def _transform_edges(
        self, other_node_id, ilocs, include_edge_weight, filter_edge_types
    ):
        if include_edge_weight:
            weights = self._edges.each_weight[ilocs]
        else:
            weights = None

        if filter_edge_types is not None:
            filter_edge_type_ilocs = self._edges.types.to_iloc(filter_edge_types)
            edge_type_ilocs = self._edges.type_ilocs[ilocs]
            correct_type = edge_type_ilocs.isin(filter_edge_type_ilocs)

            other_node_id = other_node_id[correct_type]
            if weights is not None:
                weights = weights[correct_type]

        if weights is not None:
            return [
                NeighbourWithWeight(node, weight)
                for node, weight in zip(other_node_id, weights)
            ]

        return other_node_id

    def neighbors(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring nodes.
        """
        if self._graph is not None:
            return self._graph.neighbors(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        ilocs = self._edges.edge_ilocs(node, ins=True, outs=True)
        source = self._edges.sources[ilocs]
        target = self._edges.targets[ilocs]
        other_node_id = np.where(source == node, target, source)
        return self._transform_edges(
            other_node_id, ilocs, include_edge_weight, edge_types
        )

    def in_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring in-nodes.
        """
        if self._graph is not None:
            return self._graph.in_nodes(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        ilocs = self._edges.edge_ilocs(node, ins=True, outs=False)
        source = self._edges.sources[ilocs]
        return self._transform_edges(source, ilocs, include_edge_weight, edge_types)

    def out_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.

        Returns:
            iterable: The neighbouring out-nodes.
        """
        if self._graph is not None:
            return self._graph.out_nodes(
                node, include_edge_weight=include_edge_weight, edge_types=edge_types
            )

        ilocs = self._edges.edge_ilocs(node, ins=False, outs=True)
        target = self._edges.targets[ilocs]
        return self._transform_edges(target, ilocs, include_edge_weight, edge_types)

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type (hashable, optional): a type of nodes that exist in the graph

        Returns:
            A list of node IDs with type node_type
        """
        if self._graph is not None:
            return self._graph.nodes_of_type(node_type)

        if node_type is None:
            return self.nodes()

        ilocs = self._nodes.type_range(node_type)
        return self._nodes.ids.from_iloc(locations)

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        if self._graph is not None:
            return self._graph.node_type(node)

        if not is_real_iterable(node):
            node = np.array([node])

        node_ilocs = self._nodes.ids.to_iloc(node)
        try:
            type_ilocs = self._nodes.type_ilocs[node_ilocs]
        except IndexError:
            missing_indices = np.where(self._nodes.ids.is_missing(node_ilocs))
            missing_values = comma_sep(node[missing_indices])
            raise KeyError(missing_values)

        return self._nodes.types.from_iloc(type_ilocs)

    @property
    def node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
        if self._graph is not None:
            return self._graph.node_types

        return list(self._nodes.types.pandas_index)

    def node_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types (list, optional): A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        if self._graph is not None:
            return self._graph.node_feature_sizes(node_types)

        all_sizes = self._nodes.feature_sizes()
        if node_types is None:
            return all_sizes

        node_types = set(node_types)
        return {
            type_name: size for type_name, size in all_sizes if type_name in node_types
        }

    def check_graph_for_ml(self, features=True):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        if self._graph is not None:
            return self._graph.check_graph_for_ml(features)

        if all(size == 0 for _, size in self.node_feature_sizes().items()):
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # TODO: check the schema

        # TODO: check the feature node_ids against the graph node ids?

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
        if self._graph is not None:
            return self._graph.node_features(nodes, node_type)

        node_ilocs = self._nodes.ids.to_iloc(nodes)
        if node_type is None:
            types = np.unique(self._nodes.type_of_iloc(node_ilocs))
            if len(types) == 0:
                raise ValueError(
                    "must have at least one node for inference, if `node_type` is not specified"
                )
            if len(types) > 1:
                raise ValueError("all nodes must have the same type")

            node_type = types[0]

        return self._nodes.features(node_type, node_ilocs)

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
        if self._graph is not None:
            return self._graph.info(show_attributes, sample)

        raise NotImplementedError()

    def create_graph_schema(self, create_type_maps=True, nodes=None):
        """
        Create graph schema in dict of dict format from current graph.

        Note the assumption we make that there is only one
        edge of a particular edge type per node pair.

        This means that specifying an edge by node0, node1 and edge type
        is unique.

        Arguments:
            create_type_maps (bool): If True quick lookup of node/edge types is
                created in the schema. This can be slow.

            nodes (list): A list of node IDs to use to build schema. This must
                represent all node types and all edge types in the graph.
                If specified, `create_type_maps` must be False.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema object.
        """
        if self._graph is not None:
            return self._graph.create_graph_schema(create_type_maps, nodes)

        graph_schema = {nt: set() for nt in self.node_types}
        edge_types = set()

        if nodes is None:
            selector = slice(None)
        else:
            selector = self._edges.sources.isin(nodes) & self._edges.targets.isin(nodes)

        source_ilocs = self._nodes.ids.to_iloc(self._edges.sources[selector])
        source_types = self._nodes.type_of_iloc(source_ilocs)

        rel_types = self._edges.type_of_iloc(selector)

        target_ilocs = self._nodes.ids.to_iloc(self._edges.targets[selector])
        target_types = self._nodes.type_of_iloc(target_ilocs)

        for n1, rel, n2 in zip(source_types, rel_types, target_types):
            edge_type_tri = EdgeType(n1, rel, n2)
            edge_types.add(edge_type_tri)
            graph_schema[n1].add(edge_type_tri)

            if not self.is_directed():
                edge_type_tri = EdgeType(n2, rel, n1)
                edge_types.add(edge_type_tri)
                graph_schema[n2].add(edge_type_tri)

        # Create ordered list of edge_types
        edge_types = sorted(edge_types)

        # Create keys for node and edge types
        schema = {
            node_label: sorted(node_data)
            for node_label, node_data in graph_schema.items()
        }

        return GraphSchema(
            self.is_directed(), self.node_types, edge_types, schema, None
        )

    def node_degrees(self) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        Returns:
            The degree of each node.
        """
        if self._graph is not None:
            return self._graph.node_degrees()

        degrees = self._edges.degrees()
        return dict(zip(self._nodes.ids.pandas_index, degrees))

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
        if self._graph is not None:
            return self._graph.to_adjacency_matrix(nodes)

        if nodes is None:
            index = self._nodes._id_index
            selector = slice(None)
        else:
            nodes = list(nodes)
            index = ExternalIdIndex(nodes)
            selector = self._edges.sources.isin(nodes) & self._edges.targets.isin(nodes)

        weights = self._edges.weights[selector]
        src_idx = index.to_iloc(self._edges.sources[selector])
        tgt_idx = index.to_iloc(self._edges.targets[selector])
        n = len(index)

        adj = sps.csr_matrix((self._edges.weights, (src_idx, tgt_idx)), shape=(n, n))
        if not self.is_directed:
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops
            backward = sps.csr_matrix(
                (self._edges.weights, (tgt_idx, src_idx)), shape=(n, n)
            )
            backward.setdiag(0)
            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def to_networkx(
        self,
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        edge_weight_label=globalvar.WEIGHT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
    ):
        """
        Create a NetworkX MultiGraph or MultiDiGraph instance representing this graph.

        Returns:
             An instance of `networkx.MultiDiGraph` (if directed) or `networkx.MultiGraph` (if
             undirected) containing all the nodes & edges and their types & features in this graph.
        """
        if self._graph is not None:
            return self._graph.to_networkx()

        raise NotImplementedError()

    # FIXME: Experimental/special-case methods that need to be considered more; the underscores
    # denote "package private", not fully private, and so are ok to use in the rest of stellargraph
    def _get_index_for_nodes(self, nodes, node_type=None):
        """
        Get the indices for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            n: (list or hashable) Node ID or list of node IDs
            node_type: (hashable) the type of the nodes.

        Returns:
            Numpy array containing the indices for the requested nodes.
        """
        if self._graph is not None:
            return self._graph.get_index_for_nodes(nodes, node_type)

        return self._nodes._id_index.to_iloc(nodes)

    def _adjacency_types(self, graph_schema: GraphSchema):
        """
        Obtains the edges in the form of the typed mapping:

            {edge_type_triple: {source_node: [target_node, ...]}}

        Args:
            graph_schema: The graph schema.
        Returns:
             The edge types mapping.
        """
        if self._graph is not None:
            return self._graph.adjacency_types(graph_schema)

        raise NotImplementedError()

    def _edge_weights(self, source_node: Any, target_node: Any) -> List[Any]:
        """
        Obtains the weights of edges between the given pair of nodes.

        Args:
            source_node (any): The source node.
            target_node (any): The target node.

        Returns:
            list: The edge weights.
        """
        if self._graph is not None:
            return self._graph.edge_weights(source_node, target_node)

        all_ilocs = self._edges.edge_ilocs(source_node, ins=False, outs=True)
        targets = self._edges.targets[all_ilocs]
        filtered_ilocs = all_ilocs[targets[all_ilocs] == target_node]
        return weight[filtered_ilocs]


# A convenience class that merely specifies that edges have direction.
class StellarDiGraph(StellarGraph):
    def __init__(
        self,
        graph=None,
        edge_weight_label="weight",
        node_type_name=globalvar.TYPE_ATTR_NAME,
        edge_type_name=globalvar.TYPE_ATTR_NAME,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        feature_name=globalvar.FEATURE_ATTR_NAME,
        target_name=globalvar.TARGET_ATTR_NAME,
        node_features=None,
        dtype="float32",
        nodes=None,
        edges=None,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
    ):
        super().__init__(
            graph=graph,
            is_directed=True,
            edge_weight_label=edge_weight_label,
            node_type_name=node_type_name,
            edge_type_name=edge_type_name,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            feature_name=feature_name,
            target_name=target_name,
            node_features=node_features,
            dtype=dtype,
            nodes=nodes,
            edges=edges,
            source_column=source_column,
            target_column=target_column,
        )
