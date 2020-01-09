# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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
The StellarGraph implementation that encapsulates a NetworkX graph.

"""
__all__ = ["NetworkXStellarGraph"]

from stellargraph.core.schema import EdgeType
from stellargraph.core.graph import StellarGraph

import random
import itertools as it
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
import networkx as nx

from typing import Iterable, Iterator, Any, Mapping, List, Set, Optional

from .. import globalvar
from .schema import GraphSchema
from .utils import is_real_iterable


def _convert_from_node_attribute(
    G, attr_name, node_types, node_type_name=None, node_type_default=None, dtype="f"
):
    """
    Transform the node attributes to feature vectors, for use with machine learning models.

    Each node is assumed to have a numeric array stored in the attribute_name and
    which is suitable for use in machine learning models.

    Args:
        G: NetworkX graph
        attr_name: Name of node attribute to use for conversion
        node_types: Node types in graph
        node_type_name: (optional) The name of the node attribute specifying the type.
        node_type_default: (optional) The node type of nodes without explicit type.
        dtype: (optional) The numpy datatype to create the features array.

    Returns:
        index_map: a dictionary of node_type -> {node_id: node_index}
        attribute_arrays: a dictionary of node_type -> numpy array storing the features
    """
    attribute_arrays = {}
    node_index_map = {}

    # Enumerate all nodes in graph
    nodes_by_type = {
        # XXX: This lookup does not really make sense if node_type_name is not specified - why is it optional?
        nt: [
            n
            for n, ndata in G.nodes(data=True)
            if ndata.get(node_type_name, node_type_default) == nt
        ]
        for nt in node_types
    }

    # Get the target values for each node type
    for nt in node_types:
        nt_node_list = nodes_by_type[nt]

        # Add None to node list as ID of unknown nodes
        nt_node_list.append(None)

        # Create map between node id and index (including None)
        node_index_map[nt] = {nid: ii for ii, nid in enumerate(nt_node_list)}

        # The node data
        attr_data = [
            v if v is None else G.nodes[v].get(attr_name) for v in nt_node_list
        ]

        # Get the size of the features
        data_sizes = {
            np.size(G.nodes[v].get(attr_name, []))
            for v in nt_node_list
            if v is not None
        }

        # Warn if nodes don't have the attribute
        if 0 in data_sizes:
            warnings.warn(
                "Some nodes have no value for attribute '{}', "
                "using default value.".format(attr_name),
                RuntimeWarning,
                stacklevel=2,
            )
            data_sizes.discard(0)

        # Check all are the same for this node type
        if len(data_sizes) > 1:
            raise ValueError(
                "Data sizes in nodes of type {} are inconsistent "
                "for the attribute '{}' ".format(nt, attr_name)
            )

        # If some node_type have no nodes with the attribute, skip them
        if len(data_sizes) == 0:
            continue

        # Create zero attribute array
        data_size = data_sizes.pop()

        # Dummy feature/target value for invalid nodes,
        # this will be inserted into the array in two cases:
        # 1. node ID of None (representing sampling for a missing neighbour)
        # 2. node with no attribute
        # TODO: Make these two cases more explicit, allow custom values.
        default_value = np.zeros(data_size)

        # Convert to numpy array
        attribute_arrays[nt] = np.asarray(
            [x if x is not None else default_value for x in attr_data]
        )

    return node_index_map, attribute_arrays


def _convert_from_node_data(data, node_type_map, node_types, dtype="f"):
    """
    Store the node data as feature vectors, for use with machine learning models.

    For a single node type, the data can be either:
     * a Pandas DataFrame with the index being node IDs and the columns the numeric
        feature values. Note that the features must be numeric.
     * a list or iterable of `(node_id, node_feature)` pairs where node_feature is
        a value, a list of values, or a numpy array representing the numeric feature
        values.

    For multiple node types, the data can be either:
     * a dictionary of node_type -> DataFrame with the index of each DataFrame
        being node IDs and the columns the numeric feature values.
        Note that the features must be numeric and can be different sizes for each
        node type.
     * a list or iterable of `(node_id, node_feature)` pairs where node_feature is
        a value, a list of values, or a numpy array representing the numeric feature
        values.

    Args:
        data: dict, list or DataFrame
            The data for the nodes, partitioned by node type

        node_type_map: dict
            Mapping of node_id to node_type

        node_types: list
            List of the node types in the data

        dtype: Numpy datatype optional (default='float32')
            The numpy datatype to create the features array.

    Returns:
        index_map: a dictionary of node_type -> {node_id: node_index}
        attribute_arrays: a dictionary of node_type -> numpy array storing the features
    """
    # if data is a dict of pandas dataframes or iterators, pull the features for each node type in the dictionary
    if isinstance(data, dict):
        # The keys should match the node types
        if not all(k in node_types for k in data.keys()):
            raise ValueError(
                "All node types in supplied feature dict should be in the graph"
            )

        data_arrays = {}
        data_index = {}
        for nt, arr in data.items():
            if isinstance(arr, pd.DataFrame):
                node_index_map = {nid: nii for nii, nid in enumerate(arr.index)}
                try:
                    data_arr = arr.values.astype(dtype)
                except ValueError:
                    raise ValueError(
                        "Node data passed as Pandas arrays should contain only numeric values"
                    )

            elif isinstance(arr, (Iterable, list)):
                data_arr = []
                node_index_map = {}
                for ii, (node_id, datum) in enumerate(arr):
                    data_arr.append(datum)
                    node_index_map[node_id] = ii
                data_arr = np.vstack(data_arr)

            else:
                raise TypeError(
                    "Node data should be a pandas array, an iterable, a list, or name of a node_attribute"
                )

            # Add default value to end of feature array
            default_value = np.zeros(data_arr.shape[1])
            data_arrays[nt] = np.vstack([data_arr, default_value])

            node_index_map[None] = data_arr.shape[0]
            data_index[nt] = node_index_map

    # If data is a pd.Dataframe, try pulling out the type
    elif isinstance(data, pd.DataFrame):
        if len(node_types) > 1:
            raise TypeError(
                "When there is more than one node type, pass node features as a dictionary."
            )
        node_type = next(iter(node_types))
        data_index, data_arrays = _convert_from_node_data(
            {node_type: data}, node_type_map, node_types, dtype
        )

    # If data an iterator try recreating the nodes by type
    elif isinstance(data, (Iterator, list)):
        node_data_by_type = {nt: [] for nt in node_types}
        for d in data:
            node_type = node_type_map.get(d[0])
            if node_type is None:
                raise TypeError("Node type not found in importing feature vectors!")

            node_data_by_type[node_type].append(d)

        data_index, data_arrays = _convert_from_node_data(
            node_data_by_type, node_type_map, node_types, dtype
        )

    else:
        raise TypeError(
            "Node data should be a dictionary, a pandas array, an iterable, or a tuple."
        )

    return data_index, data_arrays


class NetworkXStellarGraph(StellarGraph):
    """
    Implementation based on encapsulating a NetworkX graph.
    """

    def __init__(self, graph=None, is_directed=False, **attr):
        if is_directed:
            if not isinstance(graph, nx.MultiDiGraph):
                graph = nx.MultiDiGraph(graph)
        else:
            if not isinstance(graph, nx.MultiGraph):
                graph = nx.MultiGraph(graph)
        self._graph = graph

        # Name of optional attribute for edge weights
        self._edge_weight_label = attr.get("edge_weight_label", "weight")

        # Names of attributes that store the type of nodes and edges
        self._node_type_attr = attr.get("node_type_name", globalvar.TYPE_ATTR_NAME)
        self._edge_type_attr = attr.get("edge_type_name", globalvar.TYPE_ATTR_NAME)

        # Default types of nodes and edges
        self._node_type_default = attr.get(
            "node_type_default", globalvar.NODE_TYPE_DEFAULT
        )
        self._edge_type_default = attr.get(
            "edge_type_default", globalvar.EDGE_TYPE_DEFAULT
        )

        # Names for the feature/target type (used if they are supplied and
        #  feature/target spec not supplied"
        self._feature_attr = attr.get("feature_name", globalvar.FEATURE_ATTR_NAME)
        self._target_attr = attr.get("target_name", globalvar.TARGET_ATTR_NAME)

        # Ensure that the incoming graph data has node & edge types
        # TODO: This requires traversing all nodes and edges. Is there another way?
        node_types = set()
        type_for_node = {}
        for n, ndata in graph.nodes(data=True):
            type_for_node[n] = self._get_node_type(ndata)
            node_types.add(type_for_node[n])

        edge_types = set()
        for n1, n2, k, edata in graph.edges(keys=True, data=True):
            edge_types.add(self._get_edge_type(edata))

        # New style: we are passed numpy arrays or pandas arrays of the feature vectors
        node_features = attr.get("node_features", None)
        dtype = attr.get("dtype", "float32")

        # If node_features is a string, load features from this attribute of the nodes in the graph
        if isinstance(node_features, str):
            data_index_maps, data_arrays = _convert_from_node_attribute(
                graph,
                node_features,
                node_types,
                self._node_type_attr,
                self._node_type_default,
                dtype,
            )

        # Otherwise try importing node_features as a Numpy array or Pandas Dataframe
        elif node_features is not None:
            data_index_maps, data_arrays = _convert_from_node_data(
                node_features, type_for_node, node_types, dtype
            )

        else:
            data_index_maps = {}
            data_arrays = {}

        # TODO: What other convenience attributes do we need?
        self._nodes_by_type = None

        # This stores the feature vectors per node type as numpy arrays
        self._node_attribute_arrays = data_arrays

        # This stores the map between node ID and index in the attribute arrays
        self._node_index_maps = data_index_maps

    def __repr__(self):
        directed_str = "Directed" if self.is_directed() else "Undirected"
        s = "{}: {} multigraph\n".format(type(self).__name__, directed_str)
        s += "    Nodes: {}, Edges: {}\n".format(
            self.number_of_nodes(), self.number_of_edges()
        )
        return s

    def _get_node_type(self, node_data):
        node_type = node_data.get(self._node_type_attr)
        if node_type is None:
            node_type = self._node_type_default
            node_data[self._node_type_attr] = node_type
        return node_type

    def _get_edge_type(self, edge_data):
        edge_type = edge_data.get(self._edge_type_attr)
        if edge_type is None:
            edge_type = self._edge_type_default
            edge_data[self._edge_type_attr] = edge_type
        return edge_type

    def check_graph_for_ml(self, features=True):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        # TODO: This are simple tests and miss many problems that could arise, improve!
        # Check features on the nodes:
        if features and len(self._node_attribute_arrays) == 0:
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # TODO: check the schema

        # TODO: check the feature node_ids against the graph node ids?

    def get_index_for_nodes(self, nodes, node_type=None):
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
        if not is_real_iterable(nodes):
            nodes = [nodes]

        # Get the node type if not specified.
        if node_type is None:
            node_types = {
                self._get_node_type(self._graph.nodes[n])
                for n in nodes
                if n is not None
            }

            if len(node_types) > 1:
                raise ValueError("All nodes must be of the same type.")

            if len(node_types) == 0:
                raise ValueError(
                    "At least one node must be given if node_type not specified"
                )

            node_type = node_types.pop()

        # Get index for nodes of this type
        nt_id_to_index = self._node_index_maps[node_type]
        node_indices = [nt_id_to_index.get(n) for n in nodes]
        return node_indices

    def node_features(self, nodes, node_type=None):
        """
        Get the numeric feature vectors for the specified node or nodes.
        If the node type is not specified the node types will be found
        for all nodes. It is therefore important to supply the ``node_type``
        for this method to be fast.

        Args:
            n: (list or hashable) Node ID or list of node IDs
            node_type: (hashable) the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        # TODO: add @property decorator
        if not is_real_iterable(nodes):
            nodes = [nodes]

        # Get the node type if not specified.
        if node_type is None:
            node_types = {
                self._get_node_type(self._graph.nodes[n])
                for n in nodes
                if n is not None
            }

            if len(node_types) > 1:
                raise ValueError("All nodes must be of the same type.")

            if len(node_types) == 0:
                raise ValueError(
                    "At least one node must be given if node_type not specified"
                )

            node_type = node_types.pop()

        # Check node_types
        if (
            node_type not in self._node_attribute_arrays
            or node_type not in self._node_index_maps
        ):
            raise ValueError(f"Features not found for node type '{node_type}'")

        # Edge case: if we are given no nodes, what do we do?
        if len(nodes) == 0:
            feature_size = self._node_attribute_arrays[node_type].shape[1]
            return np.empty((0, feature_size))

        # Get index for nodes of this type
        nt_id_to_index = self._node_index_maps[node_type]
        node_indices = [nt_id_to_index.get(n) for n in nodes]

        if None in node_indices:
            problem_nodes = [
                node for node, index in zip(nodes, node_indices) if index is None
            ]
            raise ValueError(
                "Could not find features for nodes with IDs {}.".format(problem_nodes)
            )

        features = self._node_attribute_arrays[node_type][node_indices]
        return features

    def node_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types: (list) A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        # TODO: unit test!
        if not node_types:
            node_types = self.node_types

        self.check_graph_for_ml(features=True)

        fsize = {nt: self._node_attribute_arrays[nt].shape[1] for nt in node_types}
        return fsize

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type:

        Returns:
            A list of node IDs with type node_type
        """
        # TODO: unit test!
        if node_type is None:
            return list(self)
        else:
            return [
                n
                for n, ndata in self._graph.nodes(data=True)
                if self._get_node_type(ndata) == node_type
            ]

    def node_type(self, node):
        """
        Get the type of the node

        Args:
            node: Node ID

        Returns:
            Node type
        """
        return self._get_node_type(self._graph.nodes[node])

    @property
    def node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
        # TODO: unit test!
        # TODO: create a schmea when we geenrate _node_attribute_arrays and use it?
        if len(self._node_attribute_arrays) > 0:
            return set(self._node_attribute_arrays.keys())
        else:
            return {
                self._get_node_type(ndata) for n, ndata in self._graph.nodes(data=True)
            }

    def info(self, show_attributes=True, sample=None):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            sample (int): To speed up the graph analysis, use only a random sample of
                          this many nodes and edges.

        Returns:
            An information string.
        """
        directed_str = "Directed" if self.is_directed() else "Undirected"
        s = "{}: {} multigraph\n".format(type(self).__name__, directed_str)
        s += " Nodes: {}, Edges: {}\n".format(
            self.number_of_nodes(), self.number_of_edges()
        )

        # Sample the nodes for our analysis
        if sample:
            all_nodes = list(self._graph.nodes)
            snodes = random.sample(all_nodes, sample)
        else:
            snodes = None

        gs = self.create_graph_schema(create_type_maps=False, nodes=snodes)

        def is_of_edge_type(e, edge_type):
            et2 = (
                self._get_node_type(self._graph.nodes[e[0]]),
                self._get_edge_type(self._graph.edges[e]),
                self._get_node_type(self._graph.nodes[e[1]]),
            )
            return et2 == edge_type

        # Go over all node types
        s += "\n Node types:\n"
        for nt in gs.node_types:
            # Filter nodes by type
            nt_nodes = [
                ndata
                for n, ndata in self._graph.nodes(data=True)
                if self._get_node_type(ndata) == nt
            ]
            s += "  {}: [{}]\n".format(nt, len(nt_nodes))

            # Get the attributes for this node type
            attrs = set(it.chain(*[ndata.keys() for ndata in nt_nodes]))
            attrs.discard(self._node_type_attr)
            if show_attributes and len(attrs) > 0:
                s += "        Attributes: {}\n".format(attrs)

            s += "    Edge types: "
            s += ", ".join(["{}-{}->{}".format(*e) for e in gs.schema[nt]]) + "\n"

        s += "\n Edge types:\n"
        for et in gs.edge_types:
            # Filter edges by type
            et_edges = [
                e[3]
                for e in self._graph.edges(keys=True, data=True)
                if is_of_edge_type(e[:3], et)
            ]
            if len(et_edges) > 0:
                s += "    {et[0]}-{et[1]}->{et[2]}: [{len}]\n".format(
                    et=et, len=len(et_edges)
                )

            # Get the attributes for this edge type
            attrs = set(it.chain(*[edata.keys() for edata in et_edges]))
            attrs.discard(self._edge_type_attr)
            if show_attributes and len(attrs) > 0:
                s += "        Attributes: {}\n".format(attrs)

        return s

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

        if nodes is None:
            nodes = self._graph.nodes()
            edges = self._graph.edges(keys=True)

        elif create_type_maps is False:
            edges = self._graph.edges(nodes, keys=True)

        else:
            raise ValueError("Creating type maps for subsampled nodes is not supported")

        # Create node type index list
        node_types = sorted(
            {self._get_node_type(self._graph.nodes[n]) for n in nodes}, key=str
        )

        graph_schema = {nt: set() for nt in node_types}

        # Create edge type index list
        edge_types = set()
        for n1, n2, k in edges:
            edata = self._graph.adj[n1][n2][k]

            # Edge type tuple
            node_type_1 = self._get_node_type(self._graph.nodes[n1])
            node_type_2 = self._get_node_type(self._graph.nodes[n2])
            edge_type = self._get_edge_type(edata)

            # Add edge type to node_type_1 data
            edge_type_tri = EdgeType(node_type_1, edge_type, node_type_2)
            edge_types.add(edge_type_tri)
            graph_schema[node_type_1].add(edge_type_tri)

            # Also add type to node_2 data if not digraph
            if not self.is_directed():
                edge_type_tri = EdgeType(node_type_2, edge_type, node_type_1)
                edge_types.add(edge_type_tri)
                graph_schema[node_type_2].add(edge_type_tri)

        # Create ordered list of edge_types
        edge_types = sorted(edge_types)

        # Create keys for node and edge types
        schema = {
            node_label: [
                edge_types[einx]
                for einx in sorted([edge_types.index(et) for et in list(node_data)])
            ]
            for node_label, node_data in graph_schema.items()
        }

        # Create schema object
        gs = GraphSchema()
        gs._is_directed = self.is_directed()
        gs.edge_types = edge_types
        gs.node_types = node_types
        gs.schema = schema

        # Create quick type lookups for nodes and edges.
        # Note: we encode the type index, in the assumption it will take
        # less storage.
        if create_type_maps:
            node_type_map = {
                n: node_types.index(self._get_node_type(ndata))
                for n, ndata in self._graph.nodes(data=True)
            }
            edge_type_map = {
                (edge[0], edge[1], edge[2]): edge_types.index(
                    EdgeType(
                        node_types[node_type_map[edge[0]]],
                        self._get_edge_type(edge[3]),
                        node_types[node_type_map[edge[1]]],
                    )
                )
                for edge in self._graph.edges(keys=True, data=True)
            }

            gs.node_type_map = node_type_map
            gs.edge_type_map = edge_type_map

        return gs

    ######################################################################
    # Generic graph interface:

    def is_directed(self) -> bool:
        return self._graph.is_directed()

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def nodes(self) -> Iterable[Any]:
        return self._graph.nodes()

    def edges(self, triple=False) -> Iterable[Any]:
        if triple:
            # returns triples of format (node 1, node 2, edge info)
            return self._graph.edges
        else:
            # returns pairs of format (node 1, node 2)
            return self._graph.edges()

    def has_node(self, node: Any) -> bool:
        return self._graph.__contains__(node)

    def neighbors(self, node: Any) -> Iterable[Any]:
        if self.is_directed():
            in_nodes = {e[0] for e in self._graph.in_edges(node)}
            out_nodes = {e[1] for e in self._graph.out_edges(node)}
            return in_nodes | out_nodes
        return nx.neighbors(self._graph, node)

    def in_nodes(self, node: Any) -> Iterable[Any]:
        if self.is_directed():
            return {e[0] for e in self._graph.in_edges(node)}
        return nx.neighbors(self._graph, node)

    def out_nodes(self, node: Any) -> Iterable[Any]:
        if self.is_directed():
            return {e[1] for e in self._graph.out_edges(node)}
        return nx.neighbors(self._graph, node)

    ########################################################################
    # Heavy duty methods:

    def node_degrees(self) -> Mapping[Any, int]:
        return self._graph.degree()

    def to_adjacency_matrix(self, nodes: Optional[Iterable] = None):
        if nodes is not None:
            return nx.adjacency_matrix(self._graph.subgraph(nodes))
        return nx.to_scipy_sparse_matrix(
            self._graph, dtype="float32", weight=self._edge_weight_label, format="coo"
        )

    def to_networkx(self):
        # Despite this class using NetworkX, this implementation does not directly use that
        # representation, so that it can be reused as we move away from being NetworkX-based.
        if self.is_directed():
            graph = nx.MultiDiGraph()
        else:
            graph = nx.MultiGraph()

        types = self.node_types

        for ty in types:
            node_ids = self.nodes_of_type(ty)
            ty_dict = {self._node_type_attr: ty}

            if ty in self._node_attribute_arrays:
                # has features!
                features = self.node_features(node_ids, node_type=ty)

                for node_id, node_features in zip(node_ids, features):
                    graph.add_node(
                        node_id, **ty_dict, **{self._feature_attr: node_features},
                    )
            else:
                # no features, so just add the type
                graph.add_nodes_from(node_ids, **ty_dict)

        graph.add_edges_from(self.edges(triple=True))

        return graph

    # XXX This has not yet been standardised in the interface.
    def adjacency_types(self, graph_schema: GraphSchema):
        """
        Obtains the edges in the form of the typed mapping:

            {edge_type_triple: {source_node: [target_node, ...]}}

        Args:
            graph_schema: The graph schema.
        Returns:
             The edge types mapping.
        """
        edge_types = graph_schema.edge_types
        adj = {et: defaultdict(lambda: [None]) for et in edge_types}
        for n1, nbrdict in self._graph.adjacency():
            for et in edge_types:
                neigh_et = [
                    n2
                    for n2, nkeys in nbrdict.items()
                    for k in nkeys
                    if graph_schema.is_of_edge_type((n1, n2, k), et)
                ]
                # Create adjacency list in lexicographical order
                # Otherwise sampling methods will not be deterministic
                # even when the seed is set.
                adj[et][n1] = sorted(neigh_et, key=str)
        return adj

    # XXX This has not yet been standardised in the interface.
    def edge_weights(self, source_node: Any, target_node: Any) -> List[Any]:
        """
        Obtains the weights of edges between the given pair of nodes.

        Args:
            source_node (any): The source node.
            target_node (any): The target node.

        Returns:
            list: The edge weights.
        """
        edge_weight_label = self._edge_weight_label
        return [
            v.get(edge_weight_label)
            for v in self._graph[source_node][target_node].values()
        ]

    # XXX This has not yet been standardised in the interface.
    def node_attributes(self, node: Any) -> Set[Any]:
        """
        Obtains the names of any (non-standard) node attributes that are
        available in the user data.

        Args:
            node (any): The node of interest.

        Returns:
            set: The collection of node attributes.
        """
        attrs = set(self._graph.nodes[node].keys())
        # Don't use node type as attribute:
        attrs.discard(self._node_type_attr)
        return attrs
