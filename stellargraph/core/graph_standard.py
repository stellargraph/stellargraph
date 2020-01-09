# -*- coding: utf-8 -*-
#
# Copyright 2019 Data61, CSIRO
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
The standardised StellarGraph implementation.

"""
__all__ = ["StandardStellarGraph"]

from typing import Iterable, Any, Mapping, Optional, Union
import itertools

from .graph import StellarGraph
from ..data.node_data import NodeData
from ..data.edge_data import EdgeData
from ..data.edge_cache import EdgeCache
from .schema import GraphSchema, EdgeType


class StandardStellarGraph(StellarGraph):
    """
    Implementation based on encapsulating user-supplied edge and node data.
    """

    def __init__(self, is_directed=False, edges=None, nodes=None, **kwargs):
        self._is_directed = is_directed
        self._node_data = nd = self._get_node_data(nodes, kwargs)
        self._edge_data = ed = self._get_edge_data(edges, kwargs)
        self._edge_cache = EdgeCache(kwargs.get("edge_cache"), ed, nd)

    @staticmethod
    def _get_node_data(data, kwargs):
        attrs = {}
        for name in ["node_id", "node_type", "node_features", "default_node_type"]:
            value = kwargs.get(name)
            if value is not None:
                attrs[name] = value
        return NodeData(data, **attrs)

    @staticmethod
    def _get_edge_data(data, kwargs):
        attrs = {}
        for name in [
            "source_id",
            "target_id",
            "edge_id",
            "edge_type",
            "edge_weight",
            "default_edge_type",
            "default_edge_weight",
        ]:
            value = kwargs.get(name)
            if value is not None:
                attrs[name] = value
        return EdgeData(data, **attrs)

    ##################################################################
    # Common interface:

    def is_directed(self) -> bool:
        return self._is_directed

    def number_of_nodes(self) -> int:
        return self._node_data.num_nodes()

    def number_of_edges(self) -> int:
        return self._edge_data.num_edges()

    def nodes(
        self, include_info: bool = False
    ) -> Union[Iterable[Any], Iterable[tuple]]:
        return self._node_data.nodes(include_info)

    def edges(self, triple: bool = False, include_info: bool = False) -> Iterable[Any]:
        return self._edge_data.edges(triple, include_info)

    def has_node(self, node: Any) -> bool:
        return self._node_data.has_node(node)

    def neighbour_nodes(self, node: Any) -> Iterable[Any]:
        return set(self._edge_cache.neighbour_nodes(node))

    def in_nodes(self, node: Any) -> Iterable[Any]:
        if self.is_directed():
            return set(self._edge_cache.in_nodes(node))
        return self.neighbour_nodes(node)

    def out_nodes(self, node: Any) -> Iterable[Any]:
        if self.is_directed():
            return set(self._edge_cache.out_nodes(node))
        return self.neighbour_nodes(node)

    def node_type(self, node: Any) -> Any:
        return self._node_data.node_type(node)

    ##################################################################
    # Computationally intensive methods:

    def info(self, show_attributes: bool = True, sample: Optional[int] = None):
        directed_str = "Directed" if self.is_directed() else "Undirected"
        s = "{}: {} multigraph\n".format(type(self).__name__, directed_str)
        s += " Nodes: {}, Edges: {}\n".format(
            self.number_of_nodes(), self.number_of_edges()
        )
        # TODO Mimic NetworkXStellrGraph output
        return s

    def node_degrees(self) -> Mapping[Any, int]:
        raise NotImplementedError

    def adjacency_weights(self):
        raise NotImplementedError

    ##################################################################
    # Private methods:

    def check_graph_for_ml(self):
        # TODO Actually do something!!
        pass

    def create_graph_schema(
        self, create_type_maps: bool = True, nodes: Optional[Iterable[Any]] = None
    ) -> GraphSchema:
        node_data = self._node_data
        if nodes is None:
            edges = self.edges(include_info=True)
            node_types = node_data.node_type_set()
        elif create_type_maps is False:
            edges = itertools.chain(
                *[self.out_edges(n, include_info=True) for n in nodes]
            )
            node_types = {node_data.node_type(n) for n in nodes}
        else:
            raise ValueError("Creating type maps for subsampled nodes is not supported")

        # Create node type index list
        node_types = sorted(node_types)
        graph_schema = {nt: set() for nt in node_types}

        # Create edge type index list
        edge_types = set()
        for e in edges:
            # Edge type tuple
            node_type_1 = node_data.node_type(e[0])
            node_type_2 = node_data.node_type(e[1])
            edge_type = e[3]

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
            nt: [
                edge_types[einx] for einx in sorted([edge_types.index(et) for et in nd])
            ]
            for nt, nd in graph_schema.items()
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
                n[0]: node_types.index(n[1]) for n in node_data.nodes(include_info=True)
            }
            edge_type_map = {
                (edge[0], edge[1], edge[2]): edge_types.index(
                    EdgeType(edge[0], edge[3], edge[1])
                )
                for edge in edges
            }

            gs.node_type_map = node_type_map
            gs.edge_type_map = edge_type_map

        return gs

    def node_feature_sizes(self) -> Mapping[Any, int]:
        return self._node_data.node_feature_sizes()

    def node_features(self, nodes: Iterable[Any], node_type: Any = None):
        return self._node_data.node_features(nodes, node_type)
