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
A standardised interface for StellarGraph functionality.

"""
__all__ = ["StellarGraphInterface"]

from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional, Set


class StellarGraphInterface(ABC):
    """
    The standardised interface to StellarGraph objects.
    """

    @abstractmethod
    def is_directed(self) -> bool:
        """
        Indicates whether the graph is directed (True) or undirected (False).

        Returns:
             bool: The graph directedness status.
        """
        pass

    @abstractmethod
    def number_of_nodes(self) -> int:
        """
        Obtains the number of nodes in the graph.

        Returns:
             int: The number of nodes.
        """
        pass

    @abstractmethod
    def number_of_edges(self) -> int:
        """
        Obtains the number of edges in the graph.

        Returns:
             int: The number of edges.
        """
        pass

    @abstractmethod
    def nodes(self) -> Iterable[Any]:
        """
        Obtains the nodes in the graph.

        Returns:
             The collection of node identifiers.
        """
        pass

    @abstractmethod
    def edges(self) -> Iterable[tuple]:
        """
        Obtains the edges in the graph, where each edge
        is represented by a tuple containing the edge identifier
        and the source and target node identifiers.

        Returns:
             The collection of edges.
        """
        pass

    @abstractmethod
    def neighbour_nodes(self, node_id: Any) -> Set[Any]:
        """
        Obtains the set of neighbouring nodes connected
        to the given node, without regard to edge direction.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            set: The neighbouring nodes.
        """
        pass

    @abstractmethod
    def in_nodes(self, node_id: Any) -> Set[Any]:
        """
        Obtains the set of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            set: The neighbouring in-nodes.
        """
        pass

    @abstractmethod
    def out_nodes(self, node_id: Any) -> Set[Any]:
        """
        Obtains the set of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node_id (any): The identifier of the node in question.

        Returns:
            set: The neighbouring out-nodes.
        """
        pass

    @abstractmethod
    def node_features(self, nodes: Iterable[Any], node_type: Optional[Any] = None):
        """
        Obtains the numeric feature vectors for the specified nodes as a NumPy array.
        Note that any unknown node will be given a zeroed feature vector.

        Args:
            nodes (iterable): A collection of node identifiers.
            node_type (any, optional): For heterogeneous graphs, the common type of the nodes, if known.

        Returns:
            Numpy array containing the node features for the requested nodes.

        Raises:
            ValueError if the graph does not have node features.
        """
        pass
