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


class GraphSchema:
    node_types = None
    edge_types = None
    schema = None

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

    def create_type_tree(self, head_node_types, levels):
        def gen_key(key, *args):
            return key + "_".join(map(str, args))

        def get_type_tree(head_nodes, level):
            """
            Build tree in list of lists format
            Each node is a tuple of:
            (node_key, node_type, [neighbours])

            Args:
                head_nodes:
                level:

            Returns:
                List of form [ (node_key, node_type, [neighbours]), ...]
            """
            def get_neighbor_types(node_type, level, key=""):
                if level == 0:
                    return []

                neighbour_node_types = [
                    (
                        gen_key(key, ii),
                        et[2],
                        get_neighbor_types(et[2], level - 1, gen_key(key, ii) + "_"),
                    )
                    for ii, et in enumerate(self.schema[node_type])
                ]
                return neighbour_node_types

            # Create root nodes at top of heirachy and recurse in schema from head nodes.
            neighbour_node_types = [
                (
                    str(jj),
                    node_type,
                    get_neighbor_types(node_type, level, gen_key("", jj) + "#"),
                )
                for jj, node_type in enumerate(head_nodes)
            ]
            return neighbour_node_types

        def tree_to_list(tree):
            """
            Convert tree in triple-list format to a type list for HinSAGE
            Args:
                tree:
            Returns:
                List of form [(node_type, [decendents]), ...]
            """
            # Create adjacency list using BFS
            to_process = queue.Queue()
            key_dict = dict()
            clist = list()
            cinx = 0
            to_process.put(tree)
            while not to_process.empty():
                subtree = to_process.get()

                for node in subtree:
                    key, node_type, dtree = node
                    neighbours = [c[0] for c in dtree]
                    key_dict[key] = len(clist)

                    clist.append((node_type, neighbours))
                    cinx += 1
                    if len(dtree) > 0:
                        to_process.put(dtree)

            # replace keys with indices
            out_list = [(adj[0], [key_dict[x] for x in adj[1]]) for adj in clist]
            return out_list

        return tree_to_list(get_type_tree(head_node_types, levels))


class StellarGraphBase:
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # Names of attributes that store the type of nodes and edges
        self._node_type_attr = "label"
        self._edge_type_attr = "label"

    def create_graph_schema(self):
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
            edge_type_tri = (node_type_1, edge_type, node_type_2)
            edge_types.add(edge_type_tri)
            graph_schema[node_type_1].add(edge_type_tri)

            # Also add type to node_2 data if not digraph
            if not self.is_directed():
                edge_type_tri = (node_type_2, edge_type, node_type_1)
                edge_types.add(edge_type_tri)
                graph_schema[node_type_2].add(edge_type_tri)

        # Create schema object
        gs = GraphSchema()
        gs.edge_types = sorted(edge_types)
        gs.node_types = node_types

        # Create keys for node and edge types
        gs.schema = {
            node_label: [
                gs.edge_types[einx]
                for einx in sorted([gs.edge_types.index(et) for et in list(node_data)])
            ]
            for node_label, node_data in graph_schema.items()
        }
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
