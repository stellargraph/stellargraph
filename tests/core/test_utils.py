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
import numpy as np

from stellargraph.core.utils import *
from ..test_utils.graphs import example_graph_random


@pytest.fixture
def example_graph():
    return example_graph_random()


def _check_smart_index(array, idx, should_be_broadcast=False):
    result = smart_array_index(array, idx)
    np.testing.assert_array_equal(result, array[idx])

    if should_be_broadcast:
        assert result.strides[0] == 0
    else:
        assert result.strides[0] > 0


def test_smart_array_index_empty():
    d1 = np.zeros(0)
    d2 = np.zeros((0, 2))
    _check_smart_index(d1, np.array([], dtype=int))
    _check_smart_index(d2, np.array([], dtype=int))

    with pytest.raises(IndexError, match="index 0 is out of bounds"):
        smart_array_index(d1, np.array([0], dtype=int))

    with pytest.raises(IndexError, match="index -1 is out of bounds"):
        smart_array_index(d1, np.array([-1], dtype=int))

    with pytest.raises(IndexError, match="index -1 is out of bounds"):
        smart_array_index(d2, np.array([-1], dtype=int))

    with pytest.raises(IndexError, match="index -1 is out of bounds"):
        smart_array_index(d2, np.array([-1], dtype=int))


def test_smart_array_index_normal():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    _check_smart_index(arr, np.array([1]))
    _check_smart_index(arr, np.array([2, 1, 0]))
    _check_smart_index(arr, np.array([[0, 2], [0, 0]]))

    with pytest.raises(IndexError, match="index 6 is out of bounds"):
        smart_array_index(arr, np.array([6], dtype=int))

    with pytest.raises(IndexError, match="index -6 is out of bounds"):
        smart_array_index(arr, np.array([-6], dtype=int))


def test_smart_array_index_size_0():
    arr = np.zeros((3, 0, 2))
    _check_smart_index(arr, np.array([1]), should_be_broadcast=True)
    _check_smart_index(arr, np.array([2, 1, 0]), should_be_broadcast=True)
    _check_smart_index(arr, np.array([[0, 2], [0, 0]]), should_be_broadcast=True)

    # strangely enough, numpy doesn't fail for these OOB indices
    _check_smart_index(arr, np.array([6], dtype=int), should_be_broadcast=True)
    _check_smart_index(arr, np.array([-6], dtype=int), should_be_broadcast=True)


def test_smart_array_index_broadcast():
    arr = np.broadcast_to(123, (3, 4, 5))
    _check_smart_index(arr, np.array([1]), should_be_broadcast=True)
    _check_smart_index(arr, np.array([2, 1, 0]), should_be_broadcast=True)
    _check_smart_index(arr, np.array([[0, 2], [0, 0]]), should_be_broadcast=True)

    with pytest.raises(IndexError, match="index 6 is out of bounds"):
        smart_array_index(arr, np.array([6], dtype=int))

    with pytest.raises(IndexError, match="index -6 is out of bounds"):
        smart_array_index(arr, np.array([-6], dtype=int))


def _check_smart_concatenate(arrays, should_be_broadcast=False, check_strides=True):
    result = smart_array_concatenate(arrays)
    np.testing.assert_array_equal(result, np.concatenate(arrays))

    if check_strides:
        if should_be_broadcast:
            assert result.strides[0] == 0
        else:
            assert result.strides[0] > 0

    return result


def test_smart_array_concatenate_empty():
    with pytest.raises(ValueError, match="need at least one array to concatenate"):
        smart_array_concatenate([])


def test_smart_array_concatenate_normal():
    _check_smart_concatenate(
        [
            np.random.rand(0, 4, 5),
            np.random.rand(3, 4, 5),
            np.random.rand(1, 4, 5),
            np.random.rand(2, 4, 5),
        ]
    )

    _check_smart_concatenate([range(10), range(20)])

    with pytest.raises(ValueError, match="must match exactly.* size 4 .* size 6"):
        _check_smart_concatenate(
            [np.random.rand(0, 4, 5), np.random.rand(3, 6, 7),]
        )

    with pytest.raises(
        ValueError, match="must have same number.* 2 dimension.* 3 dimension"
    ):
        _check_smart_concatenate(
            [np.random.rand(0, 4), np.random.rand(3, 6, 7),]
        )


def test_smart_array_concatenate_single():
    arr = np.random.rand(3, 4, 5)
    result = _check_smart_concatenate([arr])
    assert result is arr

    # this should pass through sequences directly, because downstream might handle it better. For
    # instance, pandas.Index(range(...)) is far more efficient than pandas.Index(np.arange(...)).
    rng = range(10)
    result = _check_smart_concatenate([rng], check_strides=False)
    assert result is rng


def test_smart_array_concatenate_broadcast():
    a0 = np.broadcast_to(123, (0,))
    a1 = np.broadcast_to(123, (1,))
    a2 = np.broadcast_to(123, (2,))
    b1 = np.broadcast_to(456, (1,))
    nonbroadcast = np.array([456, 789])

    _check_smart_concatenate([a0, a1, a2], should_be_broadcast=True)
    _check_smart_concatenate([a0, b1, a2], should_be_broadcast=False)
    _check_smart_concatenate([a0, nonbroadcast, a2], should_be_broadcast=False)

    # multidimensional broadcasting isn't handled
    c43 = np.broadcast_to(123, (4, 3))
    c53 = np.broadcast_to(123, (5, 3))
    _check_smart_concatenate([c43, c53], should_be_broadcast=False)

    d43 = np.broadcast_to([123, 456, 789], (4, 3))
    d53 = np.broadcast_to([123, 456, 789], (5, 3))
    _check_smart_concatenate([d43, d53], should_be_broadcast=False)

    # weird dimensions should have normal errors
    c54 = np.broadcast_to(123, (5, 4))
    with pytest.raises(ValueError, match="must match exactly.* size 3 .* size 4"):
        _check_smart_concatenate([c43, c54])

    with pytest.raises(
        ValueError, match="must have same number.* 1 dimension.* 2 dimension"
    ):
        _check_smart_concatenate([a0, c43])


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
