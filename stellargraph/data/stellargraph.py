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
__all__ = ["GraphSchema", "StellarGraph", "StellarDiGraph", "StellarGraphBase"]

import queue
import random
import itertools as it

import pandas as pd
import numpy as np
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph

from collections import namedtuple, Iterable, Iterator

# The edge type triple
from stellargraph import globals
from stellargraph.data import utils

EdgeType = namedtuple("EdgeType", "n1 rel n2")


def _convert_from_node_attribute(
    G, attr_name, node_types, node_type_name=None, dtype="f"
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
        dtype: (optional) The numpy datatype to create the features array.

    Returns:
        index_map: a dictionary of node_type -> {node_id: node_index}
        attribute_arrays: a dictionary of node_type -> numpy array storing the features
    """
    attribute_arrays = {}
    node_index_map = {}

    # Enumerate all nodes in graph
    nodes_by_type = {
        nt: [n for n, ndata in G.nodes(data=True) if ndata[node_type_name] == nt]
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
        attr_data = [v if v is None else G.node[v].get(attr_name) for v in nt_node_list]

        # Get the size of the features
        data_sizes = {
            np.size(G.node[v].get(attr_name, [])) for v in nt_node_list if v is not None
        }

        # Warn if nodes don't have the attribute
        if 0 in data_sizes:
            print(
                "Warning: Some nodes have no value for attribute '{}', "
                "using default value.".format(attr_name)
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


class GraphSchema:
    """
    Class to encapsulate the schema information for a heterogeneous graph.

    Typically this should be created from a StellarGraph object, using the
    create_graph_schema method.
    """

    _is_directed = False
    node_types = None
    edge_types = None
    schema = None
    node_type_map = None
    edge_type_map = None

    def is_directed(self):
        return self._is_directed

    def node_index(self, name):
        """
        Return node type index from the type name

        Args:
            index: name of the node type.

        Returns:
            Numerical node type index
        """
        try:
            index = self.node_types.index(name)
        except ValueError:
            print("Warning: Node key '{}' not found.".format(name))
            index = None
        return index

    def node_index_to_type(self, index):
        """
        Return node type key from the numerical index

        Args:
            index: Numerical index of node type.

        Returns:
            Node type name
        """
        try:
            key = self.node_types[index]
        except IndexError:
            print(
                "Warning: Node index '{}' invalid. Should be an integer between 0 and {}.".format(
                    index, len(self.node_types) - 1
                )
            )
            key = None
        return key

    def edge_index(self, edge_type):
        """
        Return edge type index from the type tuple

        Args:
            index: Tuple of (node1_type, edge_type, node2_type)

        Returns:
            Numerical edge type index
        """
        try:
            index = self.edge_types.index(edge_type)
        except ValueError:
            print("Warning: Edge key '{}' not found.".format(edge_type))
            index = None
        return index

    def edge_index_to_type(self, index):
        """
        Return edge type triple from the numerical index

        Args:
            index: Numerical index of edge type.

        Returns:
            Edge type triple
        """
        try:
            key = self.edge_types[index]
        except IndexError:
            print(
                "Warning: Edge index '{}' invalid. Should be an integer between 0 and {}.".format(
                    index, len(self.edge_types) - 1
                )
            )
            key = None
        return key

    def __repr__(self):
        s = "{}:\n".format(type(self).__name__)
        for nt in self.schema:
            s += "node type: {}\n".format(nt)
            for e in self.schema[nt]:
                s += "   {} -- {} -> {}\n".format(*e)
        return s

    def get_node_type(self, node, index=False):
        # TODO: remove "get_" from the name
        """
        Return the type of the node
        Args:
            node: The node ID from the original graph
            index: Return a numeric type index if True,
                otherwise return the type name.

        Returns:
            A node type name or index
        """
        try:
            nt = self.node_type_map[node]
            node_type = nt if index else self.node_types[nt]

        except IndexError:
            print("Warning: Node '{}' not found in type map.".format(node))
            node_type = None
        return node_type

    def is_of_edge_type(self, edge, edge_type, index=False):
        """
        Tests if an edge is of the given edge type.

        The edge is specified as a standard NetworkX multigraph edge
        triple of (node_id_1, node_id_2, edge_key).

        If the graph schema is undirected then the ordering of the nodes
        of the edge type doesn't matter.

        Args:
            edge: The edge ID from the original graph as a triple.
            edge_type: The type of the edge as a tuple or EdgeType triple.

        Returns:
            True if the edge is of the given type
        """
        try:
            if edge in self.edge_type_map:
                eindex = self.edge_type_map[edge]

            elif not self.is_directed():
                eindex = self.edge_type_map[(edge[1], edge[0], edge[2])]

            else:
                raise IndexError

            et = self.edge_types[eindex]

            if self.is_directed():
                match = et == edge_type
            else:
                match = (et == edge_type) or (
                    et == (edge_type[2], edge_type[1], edge_type[0])
                )

        except IndexError:
            print("Warning: Edge '{}' not found in type map.".format(edge))
            match = False

        return match

    def get_edge_type(self, edge, index=False):
        # TODO: remove "get_" from the name
        """
        Return the type of the edge as a triple of
            (source_node_type, relation_type, dest_node_type).

        The edge is specified as a standard NetworkX multigraph edge
        triple of (node_id_1, node_id_2, edge_key).

        If the graph schema is undirected and there is an edge type for
        the edge (node_id_2, node_id_1, edge_key) then the edge type
        for this node will be returned permuted to match the node order.

        Args:
            edge: The edge ID from the original graph as a triple.
            index: Return a numeric type index if True,
                otherwise return the type triple.

        Returns:
            A node type triple or index.
        """
        try:
            if edge in self.edge_type_map:
                et = self.edge_type_map[edge]
                edge_type = et if index else self.edge_types[et]

            elif not self.is_directed():
                et = self.edge_type_map[(edge[1], edge[0], edge[2])]
                if index:
                    edge_type = et
                else:
                    et = self.edge_types[et]
                    edge_type = EdgeType(et[2], et[1], et[0])
            else:
                raise IndexError

        except IndexError:
            print("Warning: Edge '{}' not found in type map.".format(edge))
            edge_type = None
        return edge_type

    def sampling_tree(self, head_node_types, n_hops):
        """
        Returns a sampling tree for the specified head node types
        for neighbours up to n_hops away.
        A unique ID is created for each sampling node.

        Args:
            head_node_types: An iterable of the types of the head nodes
            n_hops: The number of hops away

        Returns:
            A list of the form [(type_adjacency_index, node_type, [children]), ...]
            where children are (type_adjacency_index, node_type, [children])

        """
        adjacency_list = self.type_adjacency_list(head_node_types, n_hops)

        def pack_tree(nodes, level):
            return [
                (n, adjacency_list[n][0], pack_tree(adjacency_list[n][1], level + 1))
                for n in nodes
            ]

        # The first k nodes will be the head nodes in the adjacency list
        # TODO: generalize this?
        return adjacency_list, pack_tree(range(len(head_node_types)), 0)

    def sampling_layout(self, head_node_types, num_samples):
        """
        For a sampling scheme with a list of head node types and the
        number of samples per hop, return the map from the actual
        sample index to the adjacency list index.

        Args:
            head_node_types: A list of node types of the head nodes.
            num_samples: A list of integers that are the number of neighbours
                         to sample at each hop.

        Returns:
            A list containing, for each head node type, a list consisting of
            tuples of (node_type, sampling_index). The list matches the
            list given by the method `type_adjacency_list(...)` and can be
            used to reformat the samples given by `SampledBreadthFirstWalk` to
            that expected by the HinSAGE model.
        """
        adjacency_list = self.type_adjacency_list(head_node_types, len(num_samples))
        sample_index_layout = []
        sample_inverse_layout = []

        # The head nodes are the first K nodes in the adjacency list
        # TODO: generalize this?
        for ii, hnt in enumerate(head_node_types):
            adj_to_samples = [(adj[0], []) for adj in adjacency_list]

            # The head nodes will be the first sample in the appropriate
            # sampling list, and the ii-th in the adjacenecy list
            sample_to_adj = {0: ii}
            adj_to_samples[ii][1].append(0)

            # Set the start group as the head node and point the index to the next hop
            node_groups = [(ii, hnt)]
            sample_index = 1

            # Iterate over all hops
            for jj, nsamples in enumerate(num_samples):
                next_node_groups = []
                for a_key, nt1 in node_groups:
                    # For each node we sample from all edge types from that node
                    edge_types = self.schema[nt1]

                    # We want to place the samples for these edge types in the correct
                    # place in the adjacency list
                    next_keys = adjacency_list[a_key][1]

                    for et, next_key in zip(edge_types, next_keys):
                        # These are psueo-samples for each edge type
                        sample_types = [(next_key, et.n2)] * nsamples
                        next_node_groups.extend(sample_types)

                        # Store the node type, adjacency and sampling indices
                        sample_to_adj[sample_index] = next_key
                        adj_to_samples[next_key][1].append(sample_index)
                        sample_index += 1

                        # Sanity check
                        assert adj_to_samples[next_key][0] == et.n2

                node_groups = next_node_groups

            # Add samples to layout and inverse layout
            sample_index_layout.append(sample_to_adj)
            sample_inverse_layout.append(adj_to_samples)
        return sample_inverse_layout

    def type_adjacency_list(self, head_node_types, n_hops):
        """
        Creates a BFS sampling tree as an adjacency list from head node types.

        Each list element is a tuple of:
            (node_type, [child_1, child_2, ...])
        where child_k is an index pointing to the child of the current node.

        Note that the children are ordered by edge type.

        Args:
            head_node_types: Node types of head nodes.
            n_hops: How many hops to sample.

        Returns:
            List of form [ (node_type, [children]), ...]
        """
        if not isinstance(head_node_types, (list, tuple)):
            raise TypeError("The head node types should be a list or tuple.")

        if not isinstance(n_hops, int):
            raise ValueError("n_hops should be an integer")

        to_process = queue.Queue()

        # Add head nodes
        clist = list()
        for ii, hn in enumerate(head_node_types):
            if n_hops > 0:
                to_process.put((hn, ii, 0))
            clist.append((hn, []))

        while not to_process.empty():
            # Get node, node index, and level
            nt, ninx, lvl = to_process.get()

            # The ordered list of edge types from this node type
            ets = self.schema[nt]

            # Iterate over edge types (in order)
            for et in ets:
                cinx = len(clist)
                clist.append((et.n2, []))
                clist[ninx][1].append(cinx)
                if n_hops > lvl + 1:
                    to_process.put((et.n2, cinx, lvl + 1))

        return clist


class StellarGraphBase:
    """
    StellarGraph class for undirected graph ML models. It stores both
    graph information from a NetworkX Graph object as well as features
    for machine learning.

    To create a StellarGraph object ready for machine learning, at a
    minimum pass the graph structure to the StellarGraph as a NetworkX
    graph:

    For undirected models:
    ```
    Gs = StellarGraph(nx_graph)
    ```

    For directed models:
    ```
    Gs = StellarDiGraph(nx_graph)
    ```

    To create a StellarGraph object with node features, supply the features
    as a numeric feature vector for each node.

    To take the feature vectors from a node attribute in the original NetworkX
    graph, supply the attribute name to the `node_features` argument:
    ```
    Gs = StellarGraph(nx_graph, node_features="feature")
    ```

    where the nx_graph contains nodes that have a "feature" attribute containing
    the feature vector for the node. All nodes of the same type must have
    the same size feature vectors.

    Alternatively, supply the node features as Pandas DataFrame objects with
    the of the DataFrame set to the node IDs. For graphs with a single node
    type, you can supply the DataFrame object directly to StellarGraph:
    ```
    node_data = pd.DataFrame(
        [feature_vector_1, feature_vector_2, ..],
        index=[node_id_1, node_id_2, ...]
        )
    Gs = StellarGraph(nx_graph, node_features=node_data)
    ```
    For graphs with multiple node types, provide the node features as pandas
    dataframes for each type separately, as a dictionary by node type.
    This allows node features to have different sizes for each node type:
    ```
    node_data = {
        node_type_1: pd.DataFrame(...),
        node_type_2: pd.DataFrame(...),
        }
    Gs = StellarGraph(nx_graph, node_features=node_data)
    ```

    You can also supply the node feature vectors as an iterator of `node_id`
    and feature vector pairs, for graphs with single and multiple node types:
    ```
    node_data = zip([node_id_1, node_id_2, ...], [feature_vector_1, feature_vector_2, ..])
    Gs = StellarGraph(nx_graph, node_features=node_data)
    ```

    Args:
        node_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the node types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the nodes of the graph to determine
            their type.

        edge_type_name: str, optional (default=globals.TYPE_ATTR_NAME)
            This is the name for the edge types that StellarGraph uses
            when processing heterogeneous graphs. StellarGraph will
            look for this attribute in the edges of the graph to determine
            their type.

        node_features: str, dict, list or DataFrame optional (default=None)
            This tells StellarGraph where to find the node feature information
            required by some graph models. These are expected to be
            a numeric feature vector for each node in the graph.

    """

    def __init__(self, incoming_graph_data=None, **attr):
        # TODO: add doc string
        super().__init__(incoming_graph_data, **attr)

        # Names of attributes that store the type of nodes and edges
        self._node_type_attr = attr.get("node_type_name", globals.TYPE_ATTR_NAME)
        self._edge_type_attr = attr.get("edge_type_name", globals.TYPE_ATTR_NAME)

        # Names for the feature/target type (used if they are supplied and
        #  feature/target spec not supplied"
        self._feature_attr = attr.get("feature_name", globals.FEATURE_ATTR_NAME)
        self._target_attr = attr.get("target_name", globals.TARGET_ATTR_NAME)

        # Ensure that the incoming graph data has node & edge types
        # TODO: This requires traversing all nodes and edges. Is there another way?
        # TODO: Should we add the default values as class arguments for these?
        node_types = set()
        type_for_node = {}
        for n, ndata in self.nodes(data=True):
            if self._node_type_attr not in ndata:
                ndata[self._node_type_attr] = "default"
            type_for_node[n] = ndata[self._node_type_attr]
            node_types.add(ndata[self._node_type_attr])

        edge_types = set()
        for n1, n2, k, edata in self.edges(keys=True, data=True):
            if self._edge_type_attr not in edata:
                edata[self._edge_type_attr] = "default"
            edge_types.add(edata[self._edge_type_attr])

        # New style: we are passed numpy arrays or pandas arrays of the feature vectors
        node_features = attr.get("node_features", None)
        dtype = attr.get("dtype", "float32")

        # If node_features is a string, load features from this attribute of the nodes in the graph
        if isinstance(node_features, str):
            data_index_maps, data_arrays = _convert_from_node_attribute(
                self, node_features, node_types, self._node_type_attr, dtype
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

    def fit_attribute_spec(self, *args, **kwargs):
        print("Fit attribute spec is deprecated")
        pass

    def check_graph_for_ml(self, features=True):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        # TODO: This are simple tests and miss many problems that could arise, improve!
        # TODO: At this point perhaps we should generate features rather than in fit_attribute_spec
        # TODO: but if so how do we know whether to fit the attribute specs or not?
        # Check features on the nodes:
        if features and len(self._node_attribute_arrays) == 0:
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # How about checking the schema?

    def get_feature_for_nodes(self, nodes, node_type=None):
        # TODO: change method's name to node_features(), and add @property decorator
        """
        Get the numeric feature vector for the specified node or nodes.
        If the node type is not specified the type of the first
        node in the list will be used.

        Args:
            n: Node ID or list of node IDs
            node_type: the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        if not utils.is_real_iterable(nodes):
            nodes = [nodes]

        # Get the node type
        # TODO: This is slow, refactor so that self._node_index_maps gives the node type
        if node_type is None:
            node_types = {
                self.node[n].get(self._node_type_attr) for n in nodes if n is not None
            }

            if None in node_types:
                raise ValueError(
                    "All nodes must have a type specified as the "
                    "'{}' attribute.".format(self._node_type_attr)
                )

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
            raise ValueError("Features not found for node type '{}'")

        # Edge case: if we are given no nodes, what do we do?
        if len(nodes) == 0:
            feature_size = self._node_attribute_arrays[node_type].shape[1]
            return np.empty((0, feature_size))

        # Get index for nodes of this type
        nt_id_to_index = self._node_index_maps[node_type]
        node_indices = [nt_id_to_index.get(n) for n in nodes]

        if None in node_indices:
            raise ValueError(
                "Incorrect node specified or nodes of multiple types found."
            )

        features = self._node_attribute_arrays[node_type][node_indices]
        return features

    def node_feature_sizes(self, node_types=None):
        # TODO: unit test!
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types: A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        if not node_types:
            node_types = self.node_types

        self.check_graph_for_ml(features=True)

        fsize = {nt: self._node_attribute_arrays[nt].shape[1] for nt in node_types}
        return fsize

    def nodes_of_type(self, node_type=None):
        # TODO: unit test!
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type:

        Returns:
            A list of node IDs with type node_type
        """
        if node_type is None:
            return list(self)
        else:
            return [
                n
                for n, ndata in self.nodes(data=True)
                if ndata.get(self._node_type_attr) == node_type
            ]

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
                ndata.get(self._node_type_attr) for n, ndata in self.nodes(data=True)
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
            all_nodes = list(self.nodes)
            snodes = random.sample(all_nodes, sample)
            sedges = self.edges(snodes, keys=True)
        else:
            snodes = None
            sedges = None

        gs = self.create_graph_schema(create_type_maps=True, nodes=snodes, edges=sedges)

        # Go over all node types
        s += "\n Node types:\n"
        for nt in gs.node_types:
            # Filter nodes by type
            nt_nodes = [
                ndata for n, ndata in self.nodes(data=True) if gs.get_node_type(n) == nt
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
                for e in self.edges(keys=True, data=True)
                if gs.get_edge_type(e[:3]) == et
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

    def create_graph_schema(self, create_type_maps=True, nodes=None, edges=None):
        """
        Create graph schema in dict of dict format from current graph.

        Note the assumption we make that there is only one
        edge of a particular edge type per node pair.

        This means that specifying an edge by node0, node1 and edge type
        is unique.

        Returns:
            GraphSchema object.
        """
        if nodes is None:
            nodes = self.nodes()
        elif create_type_maps is True:
            raise ValueError(
                "Creating type mapes for subsampled nodes is not supported"
            )
        if edges is None:
            edges = self.edges(keys=True)
        elif create_type_maps is True:
            raise ValueError(
                "Creating type mapes for subsampled edges is not supported"
            )

        # Create node type index list
        node_types = sorted(
            {self.node[n].get(self._node_type_attr) for n in nodes}, key=str
        )

        if None in node_types:
            raise ValueError(
                "All nodes should have a type set in the '{}' attribute.".format(
                    self._node_type_attr
                )
            )

        graph_schema = {nt: set() for nt in node_types}

        # Create edge type index list
        edge_types = set()
        for n1, n2, k in edges:
            edata = self.adj[n1][n2][k]

            # Edge type tuple
            node_type_1 = self.node[n1][self._node_type_attr]
            node_type_2 = self.node[n2][self._node_type_attr]
            edge_type = edata[self._edge_type_attr]

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
                n: node_types.index(ndata[self._node_type_attr])
                for n, ndata in self.nodes(data=True)
            }
            edge_type_map = {
                (edge[0], edge[1], edge[2]): edge_types.index(
                    EdgeType(
                        node_types[node_type_map[edge[0]]],
                        edge[3][self._edge_type_attr],
                        node_types[node_type_map[edge[1]]],
                    )
                )
                for edge in self.edges(keys=True, data=True)
            }

            gs.node_type_map = node_type_map
            gs.edge_type_map = edge_type_map

        return gs


class StellarGraph(StellarGraphBase, MultiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)


class StellarDiGraph(StellarGraphBase, MultiDiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
