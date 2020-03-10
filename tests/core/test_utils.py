# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIROß
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


"""
Utils tests:

"""
import pytest

from stellargraph.core.utils import *
from ..test_utils.graphs import example_graph_random


@pytest.fixture
def example_graph():
    return example_graph_random()


def test_normalize_adj(example_graph):
    node_list = list(example_graph.nodes())
    Aadj = example_graph.to_adjacency_matrix()
    csr = normalize_adj(Aadj, symmetric=True)
    dense = csr.todense()
    eigen_vals, _ = np.linalg.eig(dense)
    assert eigen_vals.max() == pytest.approx(1, abs=1e-7)
    assert csr.get_shape() == Aadj.get_shape()

    csr = normalize_adj(Aadj, symmetric=False)
    dense = csr.todense()
    assert 5 == pytest.approx(dense.sum(), 0.1)
    assert csr.get_shape() == Aadj.get_shape()


def test_normalized_laplacian(example_graph):
    Aadj = example_graph.to_adjacency_matrix()
    laplacian = normalized_laplacian(Aadj).todense()
    eigenvalues, _ = np.linalg.eig(laplacian)

    # min eigenvalue of normalized laplacian is 0
    # max eigenvalue of normalized laplacian is <= 2
    assert eigenvalues.min() == pytest.approx(0, abs=1e-7)
    assert eigenvalues.max() <= (2 + 1e-7)
    assert laplacian.shape == Aadj.get_shape()

    laplacian = normalized_laplacian(Aadj, symmetric=False)
    assert 1 == pytest.approx(laplacian.sum(), abs=1e-7)
    assert laplacian.get_shape() == Aadj.get_shape()


def test_rescale_laplacian(example_graph):
    node_list = list(example_graph.nodes())
    Aadj = example_graph.to_adjacency_matrix()
    rl = rescale_laplacian(normalized_laplacian(Aadj))
    assert rl.max() < 1
    assert rl.get_shape() == Aadj.get_shape()


def test_GCN_Aadj_feats_op(example_graph):
    node_list = list(example_graph.nodes())
    Aadj = example_graph.to_adjacency_matrix()
    features = example_graph.node_features(node_list)

    features_, Aadj_ = GCN_Aadj_feats_op(features=features, A=Aadj, method="gcn")
    assert np.array_equal(features, features_)
    assert 6 == pytest.approx(Aadj_.todense().sum(), 0.1)

    # k must be positive integer
    with pytest.raises(ValueError):
        GCN_Aadj_feats_op(features=features, A=Aadj, method="sgc", k=None)

    with pytest.raises(ValueError):
        GCN_Aadj_feats_op(features=features, A=Aadj, method="sgc", k=0)

    with pytest.raises(ValueError):
        GCN_Aadj_feats_op(features=features, A=Aadj, method="sgc", k=-191)

    with pytest.raises(ValueError):
        GCN_Aadj_feats_op(features=features, A=Aadj, method="sgc", k=2.0)

    features_, Aadj_ = GCN_Aadj_feats_op(features=features, A=Aadj, method="sgc", k=2)

    assert len(features_) == 6
    assert np.array_equal(features, features_)
    assert Aadj.get_shape() == Aadj_.get_shape()

    # Check if the power of the normalised adjacency matrix is calculated correctly.
    # First retrieve the normalised adjacency matrix using localpool filter.
    features_, Aadj_norm = GCN_Aadj_feats_op(features=features, A=Aadj, method="gcn")
    Aadj_norm = Aadj_norm.todense()
    Aadj_power_2 = np.linalg.matrix_power(Aadj_norm, 2)  # raise it to the power of 2
    # Both matrices should have the same shape
    assert Aadj_power_2.shape == Aadj_.get_shape()
    # and the same values.
    assert pytest.approx(Aadj_power_2) == Aadj_.todense()
