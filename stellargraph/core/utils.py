# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
import collections
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
import numpy as np


def is_real_iterable(x):
    """
    Tests if x is an iterable and is not a string.

    Args:
        x: a variable to check for whether it is an iterable

    Returns:
        True if x is an iterable (but not a string) and False otherwise
    """
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))


def zero_sized_array(shape, dtype):
    """
    Create an array with no data, without allocation.

    Args:
        shape (tuple): a shape tuple that contains at least one 0

    Returns:
        An NumPy array that contains no elements, and has a small allocation.
    """
    # FIXME: https://github.com/numpy/numpy/issues/16410
    if 0 not in shape:
        raise ValueError("shape: expected at least one zero, found {shape}")

    dtype = np.dtype(dtype)
    return np.broadcast_to(dtype.type(), shape)


def smart_array_index(array, indices):
    """
    Index array along its first dimension, smartly handling empty and broadcasted arrays to avoid
    allocating.
    """
    if len(array) > 0 and (array.size == 0 or array.strides[0] == 0):
        # this handles two cases when NumPy is suboptimal for StellarGraph:
        #
        # - 'array' containing no data, because one of its dimensions is zero length: fancy indexing
        #   allocates unnecessarily (FIXME https://github.com/numpy/numpy/issues/16410)
        #
        # - 'array' being a broadcasted array, so every element along the first dimension is the
        #   same; in some places (edge weights, in particular), it's fine to preserve this
        #   broadcasting when indexing, rather than allocating a whole new array.

        largest = indices.max(initial=0)
        smallest = indices.min(initial=0)
        valid = -len(array) <= smallest and largest < len(array)

        if valid:
            # if the indexing would work in this case, the elements are indistinguishable, so we can
            # rebroadcast (to ensure the length matches `indices`). If the indices are invalid, the
            # error is handled by the fallback below.
            final_shape = (*indices.shape, *array.shape[1:])
            return np.broadcast_to(array[:1, ...], final_shape)

    # fallback to normal indexing
    return array[indices]


def smart_array_concatenate(arrays):
    """
    Concatenate the arrays in ``arrays``, smartly handling 1D broadcasted arrays that contain all
    contain an identical value, and when ``arrays`` contains a single array.
    """
    if len(arrays) == 1:
        # concatenate allocates a new array even in this case, so we can avoid a copy
        return arrays[0]

    arrays = [np.asanyarray(arr) for arr in arrays]

    # check whether all of the arrays contain a single value, broadcasted
    nonempty = (arr for arr in arrays if len(arr) > 0)
    first = next(nonempty, None)
    if first is not None:
        element = first[0]

        def check(arr):
            return len(arr.shape) == 1 and arr.strides[0] == 0 and arr[0] == element

        if check(first) and all(check(arr) for arr in nonempty):
            # concatenate will always allocate a whole new array, which is suboptimal for edge
            # weights, where they may all be broadcasted 1s
            total_len = sum(len(arr) for arr in arrays)
            # rebroadcast the element to the final result shape
            return np.broadcast_to(element, total_len)

    # fallback to the default behaviour
    return np.concatenate(arrays)


def normalize_adj(adj, symmetric=True, add_self_loops=False):
    """
    Normalize adjacency matrix.

    Args:
        adj: adjacency matrix
        symmetric: True if symmetric normalization or False if left-only normalization
        add_self_loops: True if self loops are to be added before normalization, i.e., use A+I where A is the adjacency
            matrix and I is a square identity matrix of the same size as A.
    Returns:
        Return a sparse normalized adjacency matrix.
    """

    if add_self_loops:
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())

    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalized_laplacian(adj, symmetric=True):
    """
    Normalize graph Laplacian.

    Args:
        adj: adjacency matrix
        symmetric: True if symmetric normalization

    Returns:
        Return a normalized graph Laplacian matrix.
    """

    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def calculate_laplacian(adj):
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    return adj


def rescale_laplacian(laplacian):
    """
    Scale graph Laplacian by the largest eigenvalue of normalized graph Laplacian,
    so that the eigenvalues of the scaled Laplacian are <= 1.

    Args:
        laplacian: Laplacian matrix of the graph

    Returns:
        Return a scaled Laplacian matrix.
    """

    try:
        print("Calculating largest eigenvalue of normalized graph Laplacian...")
        largest_eigval = eigsh(laplacian, 1, which="LM", return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        warnings.warn(
            "Eigenvalue calculation did not converge! Using largest_eigval=2 instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        largest_eigval = 2

    scaled_laplacian = (2.0 / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def PPNP_Aadj_feats_op(features, A, teleport_probability=0.1):
    """
    This function calculates the personalized page rank matrix of Eq 2 in [1].
    Args:
        features: node features in the graph
        A: adjacency matrix
        teleport_probability (float): teleport probability between 0.0 and 1.0. "probability" of returning to the starting node in the
        propagation step as in [1].

    [1] `Klicpera et al., 2018 <https://arxiv.org/abs/1810.05997>`_.
    """

    if (teleport_probability > 1.0) or (teleport_probability < 0.0):
        raise ValueError(
            "teleport_probability should be between 0.0 and 1.0 (inclusive)"
        )

    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = A + sp.diags(np.ones(A.shape[0]) - A.diagonal())
    A = normalize_adj(A, symmetric=True)
    A = A.toarray()
    A = teleport_probability * np.linalg.inv(
        np.eye(A.shape[0]) - ((1 - teleport_probability) * A)
    )
    return features, A


def GCN_Aadj_feats_op(features, A, k=1, method="gcn"):

    """
    This function applies the matrix transformations on the adjacency matrix, which are required by GCN.
    GCN requires that the input adjacency matrix should be symmetric, with self-loops, and normalized.
    The features and adjacency matrix will be manipulated by either 'gcn' (applying localpool filter as a default), or
    'sgcn' filters.

    For more information about 'localpool' and 'smoothed' filters, please read details:
        [1] https://arxiv.org/abs/1609.02907
        [2] https://arxiv.org/abs/1902.07153

    Args:
        features: node features in the graph
        A: adjacency matrix
        k (int or None): If method is 'sgcn' then it should be an integer indicating the power to raise the
        normalised adjacency matrix with self loops before multiplying the node features matrix.
        method: to specify the filter to use with gcn. If method=gcn, default filter is localpool, other options are 'sgcn'.

    Returns:
        features, transformed adjacency matrix
    """

    def preprocess_adj(adj, symmetric=True):
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())
        adj = normalize_adj(adj, symmetric)
        return adj

    # build symmetric adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)

    if method == "gcn":
        # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print("Using GCN (local pooling) filters...")
        A = preprocess_adj(A)
    elif method == "chebyshev":
        raise ValueError(
            "method 'chebyshev' did not behave correctly and has been removed"
        )
    elif method == "sgc":
        # Smoothing filter (Simplifying Graph Convolutional Networks)
        if isinstance(k, int) and k > 0:
            print("Calculating {}-th power of normalized A...".format(k))
            A = preprocess_adj(A)
            A = A ** k  # return scipy.sparse.csr_matrix
        else:
            raise ValueError(
                "k should be positive integer for method='sgcn'; but received type {} with value {}.".format(
                    type(k).__name__, k
                )
            )

    return features, A
