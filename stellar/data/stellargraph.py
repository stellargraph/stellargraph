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
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph

from collections import namedtuple

# The edge type triple
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

    def _get_sampling_tree(self, head_node_types, n_hops):
        """
        Returns a sampling tree for the specified head node types
        for neighbours up to n_hops away.
        A unique ID is created for each sampling node.

        Args:
            head_node_types: An iterable of the types of the head nodes
            n_hops: The number of hops away

        Returns:
            A list of the form [(unique_id, node_type, [children]), ...]
            where children are (unique_id, edge_type, [children])

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
        self._node_type_attr = attr.get("node_type_name", "label")
        self._edge_type_attr = attr.get("edge_type_name", "label")

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

    def create_node_index_map(self):
        """
        A mapping between integer indices and node IDs and the reverse.
        This mapping is stable for graphs with the same node ids.

        Returns two mappings: `node_index_to_id` and `node_id_to_index`
        such that:
            node_index_to_id[index] = node_id
            node_id_to_index[node_id] = index

        where:
            index is an integer from 0 to G.number_of_nodes() - 1
            node_id is the label of the node in the graph,
            i.e. one of list(G)

        Returns:
            node_index_to_id, node_id_to_index
        """
        # Node IDs may be integers, strings or in fact any hashable type.
        # Convert them to strings before sorting.
        node_index_to_id = sorted(self.nodes(), key=str)

        node_id_to_index = {
            node: ii for ii,node in enumerate(node_index_to_id)
        }
        return node_index_to_id, node_id_to_index

    def info(self, sample=None):
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
        nx.to_directed
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
            if len(attrs) > 0:
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
            if len(attrs) > 0:
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
