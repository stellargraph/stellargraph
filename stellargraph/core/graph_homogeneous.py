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
An implementation of StellarGraph for a homogeneous graph specified
by Pandas data-frames for edges and nodes.

"""
__all__ = ["HomogeneousStellarGraph"]

from typing import Iterable, Any, Optional, Union, Set
import itertools

import pandas as pd
import numpy as np

from .graph_interface import StellarGraphInterface


class HomogeneousStellarGraph(StellarGraphInterface):
    """
    Creates a homogeneous graph from edge and node data (in
    Pandas data-frame format).
    """

    def __init__(
        self,
        edge_data: pd.DataFrame,
        node_data: Optional[pd.DataFrame] = None,
        is_directed: bool = False,
        edge_id: Optional[Union[int, str]] = None,
        source_id: Union[int, str] = "source",
        target_id: Union[int, str] = "target",
        node_id: Optional[Union[int, str]] = None,
        node_features: Optional[Iterable[Any]] = None,
    ):
        """
        Initialises the graph with data.

        Args:
            edge_data (data-frame): The Pandas data-frame containing edge data.
            node_data (data-frame, optional): The Pandas data-frame containing node data.
            is_directed (bool): Indicates whether the edges are directed (True)
                or undirected (False); defaults to assuming undirected edges.
            edge_id (int or str, optional): Specifies the column containing the edge identifiers;
                defaults to using the data-frame index.
            source_id (int or str, optional): Specifies the column containing the source node identifiers;
                defaults to using the "source" column.
            source_id (int or str, optional): Specifies the column containing the target node identifiers;
                defaults to using the "target" column.
            node_id (int or str, optional): Specifies the column containing the node identifiers;
                defaults to using the data-frame index.
            node_features (list of int or str, optional): Specifies the columns containing the node features.
        """
        if not isinstance(edge_data, pd.DataFrame):
            raise TypeError("Edge data must be specified in a Pandas data-frame")
        self._edges = edge_data
        if node_data is not None and not isinstance(node_data, pd.DataFrame):
            raise TypeError("Node data must be specified in a Pandas data-frame")
        self._nodes = node_data
        self._is_directed = is_directed
        self._edge_ids = self._get_column(edge_data, "edge_id", edge_id, True)
        self._src_ids = self._get_column(edge_data, "source_id", source_id)
        self._dst_ids = self._get_column(edge_data, "target_id", target_id)
        # Cache the edge nodes for faster lookup:
        # TODO Check for duplicate or missing edge identifiers
        if is_directed:
            self._out_edges = {}
            self._in_edges = {}
            for edge_id, src_id, dst_id in self.edges():
                self._out_edges.setdefault(src_id, []).append((edge_id, dst_id))
                self._in_edges.setdefault(dst_id, []).append((edge_id, src_id))
            # Check for node features and singleton nodes:
            self._node_map = node_map = {
                node_id: -1
                for node_id in itertools.chain(
                    self._out_edges.keys(), self._in_edges.keys()
                )
            }
        else:
            self._edges = {}
            for edge_id, src_id, dst_id in self.edges():
                self._edges.setdefault(src_id, []).append((edge_id, dst_id))
                if dst_id != src_id:
                    self._edges.setdefault(dst_id, []).append((edge_id, src_id))
            # Check for node features and singleton nodes:
            self._node_map = node_map = {
                node_id: -1
                for node_id in self._edges.keys()
            }
        if node_data is not None:
            node_ids = self._get_column(node_data, "node_id", node_id, True)
            for idx, node_id in enumerate(node_ids):
                # TODO Check for duplicate or missing node identifiers
                node_map[node_id] = idx
            if node_features is not None:
                self._node_features = self._get_columns(
                    node_data, "node_features", node_features
                )
            else:
                self._node_features = None
        else:
            if node_id is not None:
                raise ValueError(
                    "Cannot specify 'node_id' column if 'node_data' is None"
                )
            if node_features is not None:
                raise ValueError(
                    "Cannot specify 'node_features' columns if 'node_data' is None"
                )
            self._node_features = None

    def is_directed(self) -> bool:
        return self._is_directed

    def number_of_nodes(self) -> int:
        len(self._node_map)

    def number_of_edges(self) -> int:
        return len(self._edge_ids)

    def nodes(self) -> Iterable[Any]:
        return self._node_map.keys()

    def edges(self) -> Iterable[tuple]:
        return zip(self._edge_ids, self._src_ids, self._dst_ids)

    def neighbour_nodes(self, node_id: Any) -> Set[Any]:
        if self._is_directed:
            out_edges = self._out_edges.get(node_id, [])
            in_edges = self._in_edges.get(node_id, [])
            return {_node_id for _, _node_id in itertools.chain(out_edges, in_edges)}
        return {_node_id for _, _node_id in self._edges.get(node_id, [])}

    def in_nodes(self, node_id: Any) -> Set[Any]:
        if self._is_directed:
            return {dst_id for _, dst_id in self._in_edges.get(node_id, [])}
        return {_node_id for _, _node_id in self._edges.get(node_id, [])}

    def out_nodes(self, node_id: Any) -> Set[Any]:
        if self._is_directed:
            return {src_id for _, src_id in self._out_edges.get(node_id, [])}
        return {_node_id for _, _node_id in self._edges.get(node_id, [])}

    def node_features(self, nodes: Iterable[Any], node_type: Optional[Any] = None):
        if self._node_features is None:
            raise ValueError("The graph has no node features")
        node_idxs, special_idxs = self._extract_indices(nodes)
        if len(special_idxs) == len(node_idxs):
            # Exceptional case of all special nodes
            return np.zeros((len(node_idxs), self._node_features.shape[1]))
        feature_mat = self._node_features[node_idxs, :]
        if len(special_idxs) > 0:
            # Zero out special node features
            feature_mat[special_idxs, :] = 0
        return feature_mat

    @staticmethod
    def _get_column(df, field, column, allow_none=False):
        if column is None:
            if not allow_none:
                raise ValueError(
                    "Parameter '{}' must specify a column name or position".format(
                        field
                    )
                )
            return df.index
        if isinstance(column, int):
            column_names = list(df.columns)
            if 0 <= column < len(column_names):
                return df[column_names[column]].values
            raise ValueError(
                "Invalid column position {} for parameter '{}'".format(column, field)
            )
        if isinstance(column, str):
            if column in df.columns:
                return df[column].values
            raise ValueError(
                "Invalid column name '{}' for parameter '{}'".format(column, field)
            )
        if allow_none:
            raise TypeError(
                "Parameter '{}' must be None or specify a column name or position".format(
                    field
                )
            )
        else:
            raise TypeError(
                "Parameter '{}' must specify a column name or position".format(field)
            )

    @staticmethod
    def _get_columns(df, field, columns):
        if not isinstance(columns, Iterable):
            raise TypeError(
                "Parameter {} must specify a list of column names or positions".format(
                    field
                )
            )
        column_names = list(df.columns)
        feature_positions = []
        for i, column in enumerate(columns):
            if isinstance(column, int):
                if 0 <= column < len(column_names):
                    feature_positions.append(column)
                else:
                    raise ValueError(
                        "Invalid column position {} for parameter '{}[{}]'".format(
                            column, field, i
                        )
                    )
            elif isinstance(column, str):
                pos = column_names.index(column)
                if pos >= 0:
                    feature_positions.append(pos)
                else:
                    raise ValueError(
                        "Invalid column name {} for parameter '{}[{}]'".format(
                            column, field, i
                        )
                    )
            else:
                raise TypeError(
                    "Parameter '{}[{}]' must specify a column name or position".format(
                        field, i
                    )
                )
        return df.iloc[:, feature_positions].values

    def _extract_indices(self, node_ids):
        def lookup_fn(node_id):
            return -1 if node_id is None else self._node_map.get(node_id, -2)

        node_idxs = [0] * len(node_ids)
        special_idxs = []
        for i, node_id in enumerate(node_ids):
            node_idx = lookup_fn(node_id)
            if node_idx >= 0:
                node_idxs[i] = node_idx
            elif node_idx == -1:
                special_idxs.append(i)
            else:
                raise ValueError(
                    "The graph does not contain a node with identifier '{}'".format(
                        node_id
                    )
                )
        return node_idxs, special_idxs
