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
import queue
import random
import itertools as it

import networkx as nx
import numpy as np
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph

from collections import namedtuple

# The edge type triple
from stellar import globals
from stellar.data import utils

EdgeType = namedtuple("EdgeType", "n1 rel n2")


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

    def edge_types_for_node_type(self, node_type):
        """
        Return all edge types from a specified node type in fixed order.
        Args:
            node_type: The specified node type.

        Returns:
            A list of EdgeType instances
        """
        try:
            edge_types = self.schema[node_type]
        except IndexError:
            print("Warning: Node type '{}' not found.".format(node_type))
            edge_types = []
        return edge_types

    def get_sampling_tree(self, head_node_types, n_hops):
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
        adjacency_list = self.get_type_adjacency_list(head_node_types, n_hops)

        def pack_tree(nodes, level):
            return [
                (n, adjacency_list[n][0], pack_tree(adjacency_list[n][1], level + 1))
                for n in nodes
            ]

        # The first k nodes will be the head nodes in the adjacency list
        # TODO: generalize this?
        return adjacency_list, pack_tree(range(len(head_node_types)), 0)

    def get_sampling_layout(self, head_node_types, num_samples):
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
            list given by the method `get_type_adjacency_list(...)` and can be
            used to reformat the samples given by `SampledBreadthFirstWalk` to
            that expected by the HinSAGE model.
        """
        adjacency_list = self.get_type_adjacency_list(head_node_types, len(num_samples))
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
                    edge_types = self.edge_types_for_node_type(nt1)

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

    def get_type_adjacency_list(self, head_node_types, n_hops):
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
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # Names of attributes that store the type of nodes and edges
        self._node_type_attr = attr.get("node_type_name", globals.TYPE_ATTR_NAME)
        self._edge_type_attr = attr.get("edge_type_name", globals.TYPE_ATTR_NAME)

        # Get feature & target specifications, if supplied:
        self._feature_spec = attr.get("feature_spec", None)
        self._target_spec = attr.get("target_spec", None)

        # Names for the feature/target type (used if they are supplied and
        #  feature/target spec not supplied"
        self._feature_attr = attr.get("feature_name", globals.FEATURE_ATTR_NAME)
        self._target_attr = attr.get("target_name", globals.TARGET_ATTR_NAME)

        # These are dictionaries that store the actual feature arrays for each node type
        self._node_attribute_arrays = {}
        self._node_target_arrays = {}

        # This stores the map between node ID and index in the attribute arrays
        self._node_index_maps = {}

        # Ensure that the incoming graph data has node & edge types
        # TODO: This requires traversing all nodes and edges. Is there another way?
        for n, ndata in self.nodes(data=True):
            if self._node_type_attr not in ndata:
                ndata[self._node_type_attr] = ""

        for n1, n2, k, edata in self.edges(keys=True, data=True):
            if self._edge_type_attr not in edata:
                edata[self._edge_type_attr] = ""

    def __repr__(self):
        directed_str = "Directed" if self.is_directed() else "Undirected"
        s = "{}: {} multigraph\n".format(type(self).__name__, directed_str)
        s += "    Nodes: {}, Edges: {}\n".format(
            self.number_of_nodes(), self.number_of_edges()
        )
        return s

    def create_node_index_maps(self, schema=None, include_invalid=True):
        """
        A mapping between integer indices and node IDs and the reverse.
        This mapping is stable for graphs with the same node ids.

        Each node type has an associated `node_id_to_index` and `node_index_to_id`
        such that:
            node_index_to_id[index] = node_id
            node_id_to_index[node_id] = index

        where:
            index is an integer from 0 to number_of_nodes_for_type - 1 and is
            seperate for each node type.
            node_id is the label of the node in the graph,

        Args:
            schema: The graph schema object. If none this will be calculated
                internally.
            include_invalid: If True an invalid node with ID "None" will also
                be indexed.

        Returns:
            a dictionary with an entry for each node type with values
            being the two index maps described above, namely:
            {'type_1': (node_id_to_index, node_index_to_id), ... }

        """
        # Generate schema
        if schema is None:
            schema = self.create_graph_schema(create_type_maps=True)

        # Get the features for each node type
        node_index_maps = {}
        for nt in schema.node_types:
            nodes_for_type = [v for v in self.nodes() if schema.get_node_type(v) == nt]

            if include_invalid:
                nodes_for_type.append(None)

            # Node IDs may be integers, strings or in fact any hashable type.
            # Convert them to strings to do the sorting.
            node_index_to_id = sorted(nodes_for_type, key=str)
            node_id_to_index = {node: ii for ii, node in enumerate(node_index_to_id)}
            node_index_maps[nt] = (node_id_to_index, node_index_to_id)

        return node_index_maps

    def _convert_attributes(self, spec, node_types, attr_name, train):
        attribute_arrays = {}

        # Determine if target values will come directly from the target attribute or
        # through the target_spec converter
        if spec is None:
            # Get the target values for each node type
            for nt in node_types:
                nt_id_to_index, nt_node_list = self._node_index_maps[nt]

                attr_data = [
                    v if v is None else self.node[v].get(attr_name)
                    for v in nt_node_list
                ]

                # Get the size of the features
                data_sizes = {
                    np.size(self.node[v].get(attr_name, []))
                    for v in nt_node_list
                    if v is not None
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
                dummy_value = np.zeros(data_size)

                # Convert to numpy array
                attribute_arrays[nt] = np.asarray(
                    [x if x is not None else dummy_value for x in attr_data]
                )

        else:
            # Use feature spec to get feature vectors & put them in an array
            for nt in spec.get_types():
                nt_id_to_index, nt_node_list = self._node_index_maps[nt]

                # Convert the node attributes to a feature array
                node_data = [self.node[v] for v in nt_node_list if v is not None]
                if train:
                    aa = spec.fit_transform(nt, node_data)
                else:
                    aa = spec.transform(nt, node_data)

                # Append dummy value as final row.
                dummy_value = np.zeros(aa.shape[1], dtype=aa.dtype)
                # NOTE: This assumes that the dummy value is always the final row
                # which is how we currently construct things in _node_index_maps,
                # but is fragile
                # TODO: improve robustness here, is there a better way to do this?
                assert nt_node_list[-1] is None
                assert sum(n is None for n in nt_node_list) == 1

                attribute_arrays[nt] = np.vstack([aa, dummy_value])

        return attribute_arrays

    def set_attribute_spec(self, feature_spec=None, target_spec=None):
        """
        Transform the node attributes to feature and target vectors, for use
        with machine learning models.

        If feature_spec or target_spec are not provided, the corresponding vectors
        are assumed to be stored in the feature_name (by default "feature") and
        target_name (by default "target") attributes in the nodes and are additionally
        assumed to be suitable for use in machine learning models.

        This function is used when the feature_spec and/or target_spec have already
        been used to train a model.
        """
        self.fit_attribute_spec(feature_spec, target_spec, train=False)

    def fit_attribute_spec(self, feature_spec=None, target_spec=None, train=True):
        """
        Transform the node attributes to feature and target vectors, for use
        with machine learning models.

        If feature_spec or target_spec are not provided, the corresponding vectors
        are assumed to be stored in the feature_name (by default "feature") and
        target_name (by default "target") attributes in the nodes and are additionally
        assumed to be suitable for use in machine learning models.

        If feature_spec or target_spec is provided, the feature/target vectors will
        be created as per the specification. This is data dependant and the node
        attributes of the current state of the graph will be used to fit the supplied
        attribute specification.

        Once a machine learning model is trained, the fitted feature specifications should
        be used with that model and this function should not be used with train=True.
        Instead if using a trained machine learning model, supply the attribute
        specifications used to train that model to the `set_attribute_spec` function
        or set the flag train=False in this function.
        """
        if feature_spec is not None:
            self._feature_spec = feature_spec
        if target_spec is not None:
            self._target_spec = target_spec

        # Generate schema
        schema = self.create_graph_schema(create_type_maps=True)

        # This will store the maps between node ID and index in the attribute arrays
        self._node_index_maps = self.create_node_index_maps(schema)

        # Determine if feature attributes will come directly from the feature attribute or
        # through the feature_spec converter
        self._node_attribute_arrays = self._convert_attributes(
            self._feature_spec, schema.node_types, self._feature_attr, train=train
        )

        self._node_target_arrays = self._convert_attributes(
            self._target_spec, schema.node_types, self._target_attr, train=train
        )

        # Create graph schema at this point?

    def check_graph_for_ml(self, features=True, supervised=True):
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
                "Run 'fit_attribute_spec' on the graph with numeric feature attributes "
                "or a feature specification to generate node features for machine learning"
            )

        # Check features on the nodes:
        if supervised and len(self._node_target_arrays) == 0:
            raise RuntimeError(
                "Run 'fit_attribute_spec' on the graph with numeric target attributes "
                "or a target specification to generate node targets for supervised learning"
            )

        # How about checking the schema?

    def get_feature_for_nodes(self, nodes, node_type=None):
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
        if node_type not in self._node_attribute_arrays:
            raise ValueError(
                "Features not found for node type '{}', has fit_attribute_specs been run?"
            )

        # Edge case: if we are given no nodes, what do we do?
        if len(nodes) == 0:
            feature_size = self._node_attribute_arrays[node_type].shape[1]
            return np.empty((0, feature_size))

        # Get index for nodes of this type
        nt_id_to_index, nt_node_list = self._node_index_maps[node_type]
        node_indices = [nt_id_to_index.get(n) for n in nodes]

        if None in node_indices:
            raise ValueError(
                "Nodes specified in 'get_feature_for_nodes' "
                "must all be of the same type."
            )

        features = self._node_attribute_arrays[node_type][node_indices]
        return features

    def get_target_for_nodes(self, nodes, node_type=None):
        """
        Get the numeric target vector for the specified node or nodes
        Args:
            n: Node ID or list of node IDs

        Returns:
            List containing the node targets, if they are specified for
             the nodes, or None if the node has no target value.
        """
        if not utils.is_real_iterable(nodes):
            nodes = [nodes]

        # TODO: What if nodes are not all in graph?
        node_data = [self.node[n] for n in nodes]

        # Get the node type
        if node_type is None:
            node_types = {nd.get(self._node_type_attr) for nd in node_data}

            if None in node_types:
                raise ValueError(
                    "All nodes must have a type specified as the "
                    "'{}' attribute.".format(self._node_type_attr)
                )
            node_type = node_types.pop()

        # Check node_types
        if node_type not in self._node_target_arrays:
            raise ValueError(
                "Targets not found for node type '{}', has fit_attribute_specs been run?"
            )

        # Get index for nodes of this type
        nt_id_to_index, nt_node_list = self._node_index_maps[node_type]
        node_indices = [nt_id_to_index.get(n) for n in nodes]

        if None in node_indices:
            raise ValueError(
                "Nodes specified in 'get_target_for_nodes' "
                "must all be of the same type."
            )

        targets = self._node_target_arrays[node_type][node_indices]
        return targets

    def get_nodes_with_target(self, node_type=None):
        """
        Get the nodes that have a valid target value
        Args:
            node_type: The type label for the nodes. If None, all nodes are used.

        Returns:
            List containing the nodes that do not have a valid target attribute
        """
        nodes = self.get_nodes_of_type(node_type)

        if self._target_spec is not None:
            # If we have a target spec, find the attributes that are converted and get the nodes that don't have
            #  get the nodes that don't have one or more of these attrivutes
            target_attrs = self._target_spec.get_attributes(node_type)
            nodes_with_targets = set(nodes)
            for attr_name in target_attrs:
                nodes_with_targets.intersection_update(
                    n for n in nodes if attr_name in self.node[n]
                )
        else:
            # Otherwise directly get nodes with the _target_attr attribute
            nodes_with_targets = {n for n in nodes if self._target_attr in self.node[n]}

        # Nodes without target
        nodes_without_target = set(nodes).difference(nodes_with_targets)

        return nodes_with_targets, nodes_without_target

    def get_raw_targets(self, nodes, node_type=None, unlabeled_value=None):
        """
        Get the raw target label for the nodes, currently required for splitting.

        Args:
            node_type: The type label for the nodes. If None, all nodes are used.

        Returns:
            List containing the nodes that do not have a valid target attribute
        """
        if self._target_spec is not None:
            # If we have a target spec, find the attributes that are converted and
            #  get the nodes that don't have one or more of these attrivutes
            target_attrs = self._target_spec.get_attributes(node_type)

            # Let's just use the first one for now.
            # TODO: What happens if there are multiple targets?
            target_attr = next(iter(target_attrs))
        else:
            # Otherwise directly get nodes with the _target_attr attribute
            target_attr = self._target_attr

        target_values = [self.node[n].get(target_attr, unlabeled_value) for n in nodes]

        return target_values

    def get_feature_sizes(self, node_types=None):
        """
        Get the feature sizes for the specified node types.

        Args:
            node_types: A list of node types. If None all current node types
                will be used.

        Returns:
            A dictionary of node type and integer feature size.
        """
        # TODO: Infer node type
        self.check_graph_for_ml(features=True)

        if self._feature_spec is not None:
            node_types = self.get_node_types()
            fsize = {nt: self._feature_spec.get_output_size(nt) for nt in node_types}

        else:
            # Otherwise directly get nodes with the _target_attr attribute
            fsize = {nt: self._node_attribute_arrays[nt].shape[1] for nt in node_types}

        return fsize

    def get_feature_size(self, node_type=None):
        """
        Get the feature size for the nodes of the specified type.

        Args:
            node_type: The type label for the nodes. If None and all nodes are of
                a single type the feature size for that type is returned.

        Returns:
            The feature size is returned as an integer.
        """
        # TODO: Infer node type
        self.check_graph_for_ml(features=True)

        # TODO: Check if node type is in schema!

        if self._feature_spec is not None:
            fsize = self._feature_spec.get_output_size(node_type)

        else:
            # Otherwise directly get nodes with the _target_attr attribute
            fsize = self._node_attribute_arrays[node_type].shape[1]

        return fsize

    def get_target_size(self, node_type=None):
        """
        Get the feature size for the nodes of the specified type.

        Args:
            node_type: The type label for the nodes. If None and all nodes are of
                a single type the feature size for that type is returned.

        Returns:
            The feature size is returned as an integer.
        """
        # TODO: Infer node type
        self.check_graph_for_ml(supervised=True)

        if self._feature_spec is not None:
            fsize = self._target_spec.get_output_size(node_type)

        else:
            # Otherwise directly get nodes with the _target_attr attribute
            fsize = self._node_target_arrays[node_type].shape[1]

        return fsize

    def get_nodes_of_type(self, node_type=None):
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

    def get_node_types(self):
        """
        Get a list of all node types in the graph.

        Returns:
            set of types
        """
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
    """
    Our own class for heterogeneous undirected graphs, inherited from nx.MultiGraph,
    with extra stuff to be added that's needed by samplers and mappers
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)


class StellarDiGraph(StellarGraphBase, MultiDiGraph):
    """
    Our own class for heterogeneous directed graphs, inherited from nx.MultiDiGraph,
    with extra stuff to be added that's needed by samplers and mappers
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
