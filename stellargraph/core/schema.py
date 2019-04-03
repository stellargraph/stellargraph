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
from collections.__init__ import namedtuple
from ..core.utils import is_real_iterable

EdgeType = namedtuple("EdgeType", "n1 rel n2")


class GraphSchema:
    """
    Class to encapsulate the schema information for a heterogeneous graph.

    Typically this should be created from a StellarGraph object, using the
    :func:`~stellargraph.core.graph.create_graph_schema` method.
    """

    _is_directed = False
    node_types = None
    edge_types = None
    schema = None
    node_type_map = None
    edge_type_map = None

    def __repr__(self):
        s = "{}:\n".format(type(self).__name__)
        for nt in self.schema:
            s += "node type: {}\n".format(nt)
            for e in self.schema[nt]:
                s += "   {} -- {} -> {}\n".format(*e)
        return s

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

    def edge_index(self, edge_type):
        """
        Return edge type index from the type tuple

        Args:
            index: Tuple of (node1_type, edge_type, node2_type)

        Returns:
            Numerical edge type index
        """
        if edge_type in self.edge_types:
            index = self.edge_types.index(edge_type)

        else:
            raise ValueError("Edge key '{}' not found.".format(edge_type))

        return index

    def get_node_type(self, node, index=False):
        """
        Returns the type of the node specified either by
        node ID.

        Args:
            node: The node ID from the original graph
            index: Return a numeric type index if True,
                otherwise return the type name.

        Returns:
            A node type name or index
        """
        # TODO: deprecate this function
        if self.node_type_map is None:
            raise RuntimeError("Node type maps not enabled")

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
        # TODO: deprecate this function
        if self.edge_type_map is None:
            raise RuntimeError("Edge type maps must be created to use this method")

        if edge in self.edge_type_map:
            eindex = self.edge_type_map[edge]

        elif not self.is_directed():
            eindex = self.edge_type_map[(edge[1], edge[0], edge[2])]

        else:
            raise IndexError("Warning: Edge '{}' not found in type map.".format(edge))

        et = self.edge_types[eindex]

        if self.is_directed():
            match = et == edge_type
        else:
            match = (et == edge_type) or (
                et == (edge_type[2], edge_type[1], edge_type[0])
            )

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
        # TODO: deprecate this function
        if self.edge_type_map is None:
            raise RuntimeError("Edge type maps must be created to use this method")

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
            raise IndexError("Edge '{}' not found in type map.".format(edge))

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

        Each list element is a tuple of::

            (node_type, [child_1, child_2, ...])

        where ``child_k`` is an index pointing to the child of the current node.

        Note that the children are ordered by edge type.

        Args:
            head_node_types: Node types of head nodes.
            n_hops: How many hops to sample.

        Returns:
            List of form ``[ (node_type, [children]), ...]``
        """
        if not isinstance(head_node_types, (list, tuple)):
            raise TypeError("The head node types should be a list or tuple.")

        if not isinstance(n_hops, int):
            raise TypeError("n_hops should be an integer")

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
