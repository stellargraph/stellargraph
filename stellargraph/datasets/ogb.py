# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

from .dataset_loader import Dataset
from ..core.graph import StellarGraph, StellarDiGraph
import pandas as pd
import numpy as np


def _check_ogb():
    try:
        import ogb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e.msg}. StellarGraph can only load Open Graph Benchmark datasets using the 'ogb' module; please install it",
            name=e.name,
            path=e.path,
        ) from None


def _convert_single_graph(index, graph):
    edge_index = graph["edge_index"]
    # unused: StellarGraph doesn't (yet) support edge features
    _edge_feat = graph["edge_feat"]
    node_feat = graph["node_feat"]
    num_nodes = graph["num_nodes"]

    edges = pd.DataFrame(edge_index.T, columns=["source", "target"])
    nodes = pd.DataFrame(node_feat)

    g = StellarDiGraph(nodes, edges)
    if g.number_of_nodes() != num_nodes:
        raise AssertionError(
            f"graph {index}: expected {num_nodes} in the computed graph, found {g.number_of_nodes()}"
        )

    return g


def _load_node_dataset(name):
    _check_ogb()

    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name=name)

    splitted_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    raw_graph, labels = dataset[0]
    graph = _convert_single_graph(0, raw_graph, label)
    return graphs, labels, train_idx, valid_idx, test_idx


def _load_link_dataset(name):
    _check_ogb()

    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name=name)


def _load_graph_dataset(name):
    _check_ogb()

    from ogb.graphproppred import GraphPropPredDataset

    dataset = GraphPropPredDataset(name=name)

    splitted_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    graphs_and_labels = list(dataset)
    graphs = [
        _convert_single_graph(i, raw_graph)
        for i, (raw_graph, _) in enumerate(graphs_and_labels[:10])
    ]
    labels = np.concatenate([label for (_, label) in graphs_and_labels])
    return graphs, labels, train_idx, valid_idx, test_idx


class OgbNProteins(
    Dataset,
    name="ogbn-proteins",
    description="TODO",
    source="https://ogb.stanford.edu/docs/nodeprop/",
):
    def load(self):
        return _load_node_dataset(self.name)


class OgbNProducts(
    Dataset,
    name="ogbn-product",
    description="TODO",
    source="https://ogb.stanford.edu/docs/nodeprop/",
):
    def load(self):
        return _load_node_dataset(self.name)


class OgbGMolHIV(
    Dataset,
    name="ogbg-mol-hiv",
    description="TODO",
    source="https://ogb.stanford.edu/docs/graphprop/",
):
    def load(self):
        return _load_graph_dataset(self.name)
