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
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph

from collections import namedtuple

EdgeType = namedtuple("EdgeType", "n1 rel n2")


class GraphSchema:
    node_types = None
    edge_types = None
    schema = None
    node_type_map = None
    edge_type_map = None

    def node_key_to_index(self, key):
        try:
            index = self.node_types.index(key)
        except:
            print("Warning: Node key '{}' not found.".format(key))
            index = None
        return index

    def node_index_to_key(self, index):
        try:
            key = self.node_types[index]
        except:
            print("Warning: Node index '{}' too large.".format(index))
            key = None
        return key

    def edge_key_to_index(self, key):
        try:
            index = self.edge_types.index(key)
        except:
            print("Warning: Edge key '{}' not found.".format(key))
            index = None
        return index

    def edge_index_to_key(self, index):
        try:
            key = self.edge_types[index]
        except:
            print("Warning: Edge index '{}' too large.".format(index))
            key = None
        return key

    def __repr__(self):
        s = "{}:\n".format(type(self).__name__)
        for nt in self.schema:
            s += "node type: {}\n".format(nt)
            for e in self.schema[nt]:
                s += "   {} -- {} -> {}\n".format(*e)
        return s

    def node_type(self, node, index=False):
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
            node_type = nt if index else self.node_types[index]

        except:
            print("Warning: Node '{}' not found in type map.".format(node))
            node_type = None
        return node_type

    def edge_type(self, edge, index=False):
        """
        Return the type of the edge
        Args:
            edge: The edge ID from the original graph [a tuple (n1,n2)]
            index: Return a numeric type index if True,
                otherwise return the type triple:
                 (source_node_type, relation_type, dest_node_type).

        Returns:
            A node type triple or index
        """
        try:
            if edge in self.edge_type_map:
                et = self.edge_type_map[edge]
            else:
                et = self.edge_type_map[(edge[1], edge[0])]
            edge_type = et if index else self.edge_types[et]

        except:
            print("Warning: Edge '{}' not found in type map.".format(edge))
            edge_type = None
        return edge_type

    def edge_types_for_node(self, node_type):
        try:
            edge_types = self.schema[node_type]
        except:
            print("Warning: Node type '{}' not found.".format(node_type))
            edge_types = []
        return edge_types

    def create_sampling_tree(self, head_node_types, n_hops):
        """
        Returns a sampling tree for the specified head node types
        for neighbours up to n_hops away.

        Args:
            head_node_types: An iterable of the types of the head nodes
            n_hops: The number of hops away

        Returns:

        """

        def gen_key(key, *args):
            return key + "_".join(map(str, args))

        def get_neighbor_types(node_type, level, key=""):
            if level == 0:
                return []

            neighbour_node_types = [
                (
                    gen_key(key, ii),
                    et,
                    get_neighbor_types(et.n2, level - 1, gen_key(key, ii) + "_"),
                )
                for ii, et in enumerate(self.schema[node_type])
            ]
            return neighbour_node_types

        # Create root nodes at top of heirachy and recurse in schema from head nodes.
        neighbour_node_types = [
            (
                str(jj),
                node_type,
                get_neighbor_types(node_type, n_hops, gen_key("", jj) + "#"),
            )
            for jj, node_type in enumerate(head_node_types)
        ]
        return neighbour_node_types

    def create_type_tree(self, head_node_types, n_hops):
        """
        Creates a BFS sampling tree in list of lists format from head node types.
        Each node is a tuple of:
        (node_type, [children])

        Args:
            head_node_types: Node types of head nodes.
            n_hops: How many hops to sample.

        Returns:
            List of form [ (node_type, [children]), ...]
        """
        to_process = queue.Queue()

        # Add head nodes
        clist = list()
        for ii, hn in enumerate(head_node_types):
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
                if lvl < n_hops:
                    to_process.put((et.n2, cinx, lvl + 1))

        return clist


class StellarGraphBase:
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # Names of attributes that store the type of nodes and edges
        self._node_type_attr = attr.get("node_type_key", "label")
        self._edge_type_attr = attr.get("edge_type_key", "label")

    def create_graph_schema(self, create_type_maps=True):
        """
        Create graph schema in dict of dict format from current graph
        Returns:
            GraphSchema object.
        """
        # Create node type index list
        node_types = sorted(
            {ndata[self._node_type_attr] for n, ndata in self.nodes(data=True)}
        )
        graph_schema = {nt: set() for nt in node_types}

        # Create edge type index list
        edge_types = set()
        for n1, n2, edata in self.edges(data=True):
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
        edge_types =  sorted(edge_types)

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
                (n1, n2): edge_types.index(
                    EdgeType(
                        node_types[node_type_map[n1]],
                        edata[self._edge_type_attr],
                        node_types[node_type_map[n2]],
                    )
                )
                for n1, n2, edata in self.edges(data=True)
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
