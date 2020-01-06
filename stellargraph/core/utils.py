# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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
    return isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes))


def normalize_adj(adj, symmetric=True):
    """
    Normalize adjacency matrix.

    Args:
        adj: adjacency matrix
        symmetric: True if symmetric normalization or False if left-only normalization

    Returns:
        Return a sparse normalized adjacency matrix.
    """

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
        print(
            "Eigenvalue calculation did not converge! Using largest_eigval=2 instead."
        )
        largest_eigval = 2

    scaled_laplacian = (2.0 / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """
    Calculate Chebyshev polynomials up to order k. For more info, see https://en.wikipedia.org/wiki/Chebyshev_filter

    Args:
        X: adjacency matrix
        k: maximum polynomial degree

    Returns:
        Return a list of sparse matrices.
    """

    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


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
    The features and adjacency matrix will be manipulated by either 'gcn' (applying localpool filter as a default), 'chebyshev', or
    'sgcn' filters.

    For more information about 'localpool', 'chebyshev', and 'smoothed' filters, please read details:
        [1] https://en.wikipedia.org/wiki/Chebyshev_filter
        [2] https://arxiv.org/abs/1609.02907
        [3] https://arxiv.org/abs/1902.07153

    Args:
        features: node features in the graph
        A: adjacency matrix
        k (int or None): If method is 'sgcn' then it should be an integer indicating the power to raise the
        normalised adjacency matrix with self loops before multiplying the node features matrix.
        If method is 'chebyshev' then it should be an integer indicating the maximum order of the Chebyshev polynomials.
        method: to specify the filter to use with gcn. If method=gcn, default filter is localpool, other options are 'chebyshev' and 'sgcn'.


    Returns:
        features (transformed in case of "chebyshev" filter applied), transformed adjacency matrix
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
        # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)
        print("Using Chebyshev polynomial filters...")
        if isinstance(k, int) and k >= 2:
            # default minimum degree for Chebyshev polynomials.
            T_k = chebyshev_polynomial(rescale_laplacian(normalized_laplacian(A)), k)
            features = [features] + T_k
        else:
            raise ValueError(
                "max_degree should be positive integer of value at least 2 for method='chebyshev'; but received type {} with value {}.".format(
                    type(k), k
                )
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
                    type(k), k
                )
            )

    return features, A
