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
from stellar import GLOBALS

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
        self._node_type_attr = attr.get("node_type_name", GLOBALS.TYPE_ATTR_NAME)
        self._edge_type_attr = attr.get("edge_type_name", GLOBALS.TYPE_ATTR_NAME)

        # Get feature & target specifications, if supplied:
        self._feature_spec = attr.get("feature_spec", None)
        self._target_spec = attr.get("target_spec", None)

        # Names for the feature/target type (used if they are supplied and
        #  feature/target spec not supplied"
        self._feature_attr = attr.get("feature_name", GLOBALS.FEATURE_ATTR_NAME)
        self._feature_attr = attr.get("target_name", GLOBALS.TARGET_ATTR_NAME)

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

    def create_node_index_maps(self, schema=None):
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

            # Node IDs may be integers, strings or in fact any hashable type.
            # Convert them to strings to do the sorting.
            node_index_to_id = sorted(nodes_for_type, key=str)
            node_id_to_index = {node: ii for ii, node in enumerate(node_index_to_id)}
            node_index_maps[nt] = (node_id_to_index, node_index_to_id)

        return node_index_maps

    def convert_attributes_to_numeric(self):
        """
        This is run automatically by the ML components of stellar. It converts the attributes
        specified in the NodeAttributeSpecification object for features and targets to
        numeric vectors.
        """
        # Create feature array dictionary
        # This will store the feature arrays for each type of node
        self._node_attribute_arrays = {}

        # Generate schema
        schema = self.create_graph_schema(create_type_maps=True)

        # This will store the maps between node ID and index in the attribute arrays
        self._node_index_maps = self.create_node_index_maps(schema)

        # Determine if feature attributes will come directly from the feature attribute or
        # through the feature_spec converter
        if self._feature_spec is not None:
            # Get the features for each node type
            for nt in schema.node_types:
                nt_id_to_index, nt_node_list = self._node_index_maps[nt]

                feature_data = [
                    self.node[v].get(self._feature_attr) for v in nt_node_list
                ]
                # Check the if there are nodes without features
                if None in feature_data:
                    raise RuntimeError(
                        "Not all nodes are required to have a numeric feature "
                        "vector this should be in the attribute named '{}'".format(
                            self._feature_attr
                        )
                    )

                # Get the size of the features
                n_nodes = len(feature_data)
                feature_sizes = {np.size(a) for a in feature_data}
                # Check all are the same for this node type
                if len(feature_sizes) > 1:
                    raise ValueError(
                        "Feature sizes in nodes of type {} is inconsistent, "
                        "found the following feature sizes: {}".format(
                            nt, feature_sizes
                        )
                    )

                # Convert to numpy array
                self._node_attribute_arrays[nt] = np.asanyarray(feature_data)

        else:
            # Use feature spec to get feature vectors & put them in an array
            for nt in schema.node_types:
                nt_id_to_index, nt_node_list = self._node_index_maps[nt]
                n_nodes = len(nt_node_list)

                # Convert the node attributes to a feature array
                node_data = [self.node[v] for v in nt_node_list]
                self._node_attribute_arrays[nt] = self._feature_spec.fit_transform_all(nt, node_data)

    def get_feature_for_nodes(self, nodes):
        """
        Get the numeric feature vector for the specified node or nodes
        Args:
            n: Node ID or list of node IDs

        Returns:
            Numpy array containing the node features for the requested nodes.
        """
        pass

    def get_target_for_nodes(self, nodes):
        """
        Get the numeric target vector for the specified node or nodes
        Args:
            n: Node ID or list of node IDs

        Returns:
            List containing the node targets, if they are specified for
             the nodes, or None if the node has no target value.
        """
        pass

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
        node_types = sorted({self.node[n][self._node_type_attr] for n in nodes})
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
