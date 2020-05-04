# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tempfile
import os
import numpy as np
from stellargraph.datasets import *
from urllib.error import URLError
from stellargraph.datasets.dataset_loader import DatasetLoader
from urllib.request import urlretrieve
from unittest.mock import patch


# use parametrize to automatically test each of the datasets that (directly) derive from DatasetLoader
@pytest.mark.parametrize("dataset_class", list(DatasetLoader.__subclasses__()))
def test_dataset_download(dataset_class):
    dataset_class().download(ignore_cache=True)


@patch(
    "stellargraph.datasets.datasets.Cora.url", new="http://stellargraph-invalid-url/x"
)
def test_invalid_url() -> None:
    with pytest.raises(URLError):
        Cora().download(ignore_cache=True)


# we add an additional expected file that should break the download
@patch(
    "stellargraph.datasets.datasets.Cora.expected_files",
    new=Cora.expected_files + ["test-missing-file.xyz"],
)
def test_missing_files() -> None:
    # download - the url should work, but the files extracted won't be correct
    with pytest.raises(FileNotFoundError):
        Cora().download()


def test_environment_path_override(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as new_datasets_path:
        monkeypatch.setenv("STELLARGRAPH_DATASETS_PATH", new_datasets_path)
        dataset = CiteSeer()
        assert dataset.base_directory == os.path.join(
            new_datasets_path, dataset.directory_name
        )
        dataset.download()


@patch("stellargraph.datasets.dataset_loader.urlretrieve", wraps=urlretrieve)
def test_download_cache(mock_urlretrieve) -> None:
    # forcing a re-download should call urlretrieve
    Cora().download(ignore_cache=True)
    assert mock_urlretrieve.called

    mock_urlretrieve.reset_mock()

    # if already downloaded and in the cache, then another download should skip urlretrieve
    Cora().download()
    assert not mock_urlretrieve.called


def test_blogcatalog3_load() -> None:
    g = BlogCatalog3().load()

    n_users = 10312
    n_groups = 39
    n_friendships = 333983
    n_belongs_to = 14476

    assert g.number_of_nodes() == n_users + n_groups
    assert g.number_of_edges() == n_friendships + n_belongs_to

    assert list(g.nodes(node_type="user")) == [f"u{x}" for x in range(1, n_users + 1)]
    assert list(g.nodes(node_type="group")) == [f"g{x}" for x in range(1, n_groups + 1)]


def _graph_kernels_load(
    dataset,
    n_graphs,
    total_nodes,
    max_nodes,
    mean_nodes,
    total_edges,
    mean_edges,
    expected_labels,
    node_features,
):
    graphs, labels = dataset.load()

    assert len(graphs) == n_graphs
    assert len(labels) == n_graphs

    n_nodes = [g.number_of_nodes() for g in graphs]
    assert sum(n_nodes) == total_nodes
    assert max(n_nodes) == max_nodes

    n_edges = [g.number_of_edges() for g in graphs]
    assert sum(n_edges) == total_edges

    # verify that the numbers we've written match the averages from the table in
    # https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
    assert np.mean(n_nodes) == pytest.approx(mean_nodes, 0.005)
    # directed -> undirected doubles the number of edges
    assert np.mean(n_edges) == pytest.approx(mean_edges * 2, 0.005)

    assert set(labels) == expected_labels

    # all graphs should be homogeneous
    node_labels = {tuple(g.node_types) for g in graphs}
    assert node_labels == {("default",)}

    feature_sizes = {g.node_feature_sizes()["default"] for g in graphs}
    assert feature_sizes == {node_features}


def test_mutag_load() -> None:
    _graph_kernels_load(
        MUTAG(),
        n_graphs=188,
        total_nodes=3371,
        max_nodes=28,  # graph 6
        total_edges=7442,
        expected_labels={"-1", "1"},
        node_features=7,  # 7 labels
        mean_nodes=17.93,
        mean_edges=19.79,
    )


def test_proteins_load() -> None:
    _graph_kernels_load(
        PROTEINS(),
        n_graphs=1113,
        total_nodes=43471,
        max_nodes=620,  # graph 77
        total_edges=162088,
        expected_labels={"1", "2"},
        node_features=3 + 1,  # 3 labels, one attribute
        mean_nodes=39.06,
        mean_edges=72.82,
    )


def test_movielens_load() -> None:
    g, edges_with_ratings = MovieLens().load()

    n_users = 943
    n_movies = 1682
    n_ratings = 100000

    assert g.number_of_nodes() == n_users + n_movies
    assert g.number_of_edges() == n_ratings

    assert len(g.nodes(node_type="user")) == n_users
    assert len(g.nodes(node_type="movie")) == n_movies

    assert len(edges_with_ratings) == n_ratings
    assert list(edges_with_ratings.columns) == ["user_id", "movie_id", "rating"]


@pytest.mark.parametrize("is_directed", [False, True])
@pytest.mark.parametrize("largest_cc_only", [False, True])
@pytest.mark.parametrize("subject_as_feature", [False, True])
def test_cora_load(is_directed, largest_cc_only, subject_as_feature) -> None:
    g, subjects = Cora().load(is_directed, largest_cc_only, subject_as_feature)

    if largest_cc_only:
        expected_nodes = 2485
        expected_edges = 5209
    else:
        expected_nodes = 2708
        expected_edges = 5429

    base_feature_size = 1433
    if subject_as_feature:
        feature_size = base_feature_size + 7
    else:
        feature_size = base_feature_size

    assert g.nodes().dtype == int
    assert g.is_directed() == is_directed

    assert g.number_of_nodes() == expected_nodes
    assert g.number_of_edges() == expected_edges
    assert g.node_feature_sizes() == {"paper": feature_size}

    assert len(subjects) == g.number_of_nodes()
    assert set(subjects.index) == set(g.nodes())
    assert set(subjects) == {
        "Case_Based",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Reinforcement_Learning",
        "Rule_Learning",
        "Theory",
    }


def test_cora_load_weighted() -> None:
    def weights(graph, subjects, edges):
        sources = graph.node_features(edges.source)
        targets = graph.node_features(edges.target)

        and_ = np.logical_and(sources, targets).sum(axis=1)
        or_ = np.logical_or(sources, targets).sum(axis=1)
        jaccard = and_ / or_

        same_subject = (
            subjects[edges.source].to_numpy() == subjects[edges.target].to_numpy()
        )

        return same_subject + jaccard

    g, subjects = Cora().load(edge_weights=weights)

    _, weights = g.edges(include_edge_weight=True)
    # some edges have neither subject nor any features in common
    assert weights.min() == 0.0
    # "same subject" is either 0 or 1 and some edges definitely have 1, and jaccard is in [0, 1], so
    # we can get a bound on the weights:
    assert 1 <= weights.max() <= 2


def test_cora_load_str() -> None:
    g, subjects = Cora().load(str_node_ids=True)

    # if everything is wrong, a top-level == gives better errors
    assert type(g.nodes()[0]) == str
    # but still good to check everything
    assert all(type(n) == str for n in g.nodes())

    assert set(subjects.index) == set(g.nodes())


def test_aifb_load() -> None:
    g, affiliation = AIFB().load()

    assert g.number_of_nodes() == 8285
    assert g.number_of_edges() == 29043
    # 'affiliation' and 'employs' are excluded
    assert len(set(et for _, _, et in g.edges(include_edge_type=True))) == 47 - 2
    assert g.node_feature_sizes() == {"default": 8285}

    assert len(affiliation) == 178


@pytest.mark.parametrize("largest_cc_only", [False, True])
def test_citeseer_load(largest_cc_only) -> None:
    g, subjects = CiteSeer().load(largest_cc_only)

    if largest_cc_only:
        expected_nodes = 2110
        expected_edges = 3757
    else:
        expected_nodes = 3312
        expected_edges = 4715

    assert g.number_of_nodes() == expected_nodes
    assert g.number_of_edges() == expected_edges

    assert len(subjects) == g.number_of_nodes()
    assert set(subjects.index) == set(g.nodes())

    assert set(subjects) == {"AI", "Agents", "DB", "HCI", "IR", "ML"}


def _knowledge_graph_load(dataset, nodes, rels, train, test, valid):
    g, train_df, test_df, valid_df = dataset.load()

    assert g.number_of_nodes() == nodes
    assert g.number_of_edges() == train + test + valid
    assert len({et for _, _, et in g.edges(include_edge_type=True)}) == rels

    assert len(train_df) == train
    assert len(test_df) == test
    assert len(valid_df) == valid

    cols = {"source", "label", "target"}
    assert set(train_df.columns) == cols
    assert set(test_df.columns) == cols
    assert set(valid_df.columns) == cols


def test_wn18_load() -> None:
    _knowledge_graph_load(
        WN18(), nodes=40943, rels=18, train=141442, test=5000, valid=5000,
    )


def test_wn18rr_load() -> None:
    _knowledge_graph_load(
        WN18RR(), nodes=40943, rels=11, train=86835, test=3134, valid=3034,
    )


def test_fb15k_load() -> None:
    _knowledge_graph_load(
        FB15k(), nodes=14951, rels=1345, train=483142, test=59071, valid=50000,
    )


def test_fb15k_237_load() -> None:
    _knowledge_graph_load(
        FB15k_237(), nodes=14541, rels=237, train=272115, test=20466, valid=17535,
    )


def test_pubmeddiabetes_load() -> None:
    g, labels = PubMedDiabetes().load()

    n_nodes = 19717

    assert g.number_of_nodes() == n_nodes
    assert g.number_of_edges() == 44338

    assert g.node_feature_sizes() == {"paper": 500}

    assert len(labels) == n_nodes
    assert set(labels.index) == set(g.nodes())


def test_ia_enron_employees_load() -> None:
    graph, edges = IAEnronEmployees().load()

    n_nodes = 151
    n_edges = 50572

    assert graph.number_of_nodes() == n_nodes
    assert graph.number_of_edges() == n_edges
    assert len(edges) == n_edges
    assert set(edges.columns) == {"source", "target", "time"}
