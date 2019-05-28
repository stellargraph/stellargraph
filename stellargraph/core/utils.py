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


#def GCN_Aadj_feats_op(features, A, k=1, **kwargs):
def GCN_Aadj_feats_op(features, A, k=1, method="gcn", max_degree=2):

    """
    This function applies the matrix transformations on the adjacency matrix, which are required by GCN.
    GCN requires that the input adjacency matrix should be symmetric, with self-loops, and normalized.
    The features and adjacency matrix will be manipulated by either 'localpool', 'chebyshev', or
    'smoothed' filters.

    For more information about 'localpool', 'chebyshev', and 'smoothed' filters, please read details:
        [1] https://en.wikipedia.org/wiki/Chebyshev_filter
        [2] https://arxiv.org/abs/1609.02907
        [3] https://arxiv.org/abs/1902.07153

    Args:
        features: node features in the graph
        A: adjacency matrix
        k (int or None): If filter is 'smoothed' then it should be an integer indicating the power to raise the
        normalised adjacency matrix with self loops before multiplying the node features matrix.
        kwargs: additional arguments for choosing filter: localpool, chebyshev, or smoothed
                (For example, pass filter="localpool" as an additional argument to apply the localpool filter)

    Returns:
        features (transformed in case of "chebyshev" filter applied), transformed adjacency matrix
    """

    def preprocess_adj(adj, symmetric=True):
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_adj(adj, symmetric)
        return adj

    # build symmetric adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    #filter = kwargs.get("filter", "localpool")

    filter = method
    
    if filter == "gcn":
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print("Using local pooling filters...")
        A = preprocess_adj(A)
    elif filter == "chebyshev":
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print("Using Chebyshev polynomial basis filters...")
        #max_degree = kwargs.get("max_degree", 2)
        if isinstance(max_degree, int) and k > 0:
            T_k = chebyshev_polynomial(
                    rescale_laplacian(normalized_laplacian(A)), max_degree
                    )
            features = [features] + T_k
        else:
            raise ValueError(
                "max_degree should be positive integer for filter='chebyshev'; but received type {} with value {}.".format(
                    type(max_degree), max_degree
                )
            )
   
    elif filter == "sgcn":
        """ Smoothing filter (Simplifying Graph Convolutional Networks) """
        if isinstance(k, int) and k > 0:
            print("Calculating {}-th power of normalized A...".format(k))
            A = preprocess_adj(A)
            A = A ** k  # return scipy.sparse.csr_matrix
        else:
            raise ValueError(
                "k should be positive integer for filter='smoothed'; but received type {} with value {}.".format(
                    type(k), k
                )
            )

    return features, A
