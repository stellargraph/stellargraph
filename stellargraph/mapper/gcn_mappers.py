from keras.utils import Sequence
from keras import Input
import scipy.sparse as sp
import numpy as np
import networkx as nx

from ..core.utils import is_real_iterable
from ..core.graph import StellarGraphBase, GraphSchema

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


class FullBatchNodeSequence(Sequence):
    def __init__(self, features, A, support, suppG, node_indices, targets=None, sample_weight=None):
        # Check targets is iterable & has the correct length
        if not is_real_iterable(targets):
            raise TypeError("Targets must be None or an iterable or numpy array ")

        self.features = features
        self.A = A
        self.support = support
        self.suppG = suppG
        self.node_ind = node_indices
        self.targets = targets
        self.sample_weight = sample_weight

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return [self.features, self.A], self.targets, self.sample_weight

class FullBatchNodeGenerator:
    def __init__(self, G, name=None, filter='localpool', max_degree=2):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml()

        self.name = name
        self.graph = G
        self.nodes = list(G.nodes())

        self.features = G.get_feature_for_nodes(self.nodes)
        self.max_degree = max_degree

        self.A = nx.adjacency_matrix(G, nodelist=self.nodes)

        # build symmetric adjacency matrix
        self.A = self.A + self.A.T.multiply(self.A.T > self.A) - self.A.multiply(self.A.T > self.A)

        if filter == 'localpool':
            """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
            print('Using local pooling filters...')
            # self.A = preprocess_adj(self.A)
            self.support = 1
            self.suppG = [Input(shape=(None,None), batch_shape=(None,None), sparse=True)]
        elif filter == 'chebyshev':
            """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
            print('Using Chebyshev polynomial basis filters...')
            T_k = chebyshev_polynomial(rescale_laplacian(normalized_laplacian(self.A)), 2)
            self.features = [self.features]+T_k
            self.support = self.max_degree + 1
            self.suppG = [Input(shape=(None,None), batch_shape=(None,None), sparse=True) for _ in range(support)]

    def flow(self, node_ids, targets=None):
        node_indices = [self.nodes.index(n) for n in node_ids]

        if targets is not None:
            node_mask = np.zeros(len(self.nodes), dtype=int)
            node_mask[node_indices] = 1
            node_mask = np.array(node_mask, dtype=np.bool)
            y = np.zeros( (len(self.nodes),targets.shape[1]), dtype=np.int32)
            for idx, t in zip(node_indices, targets):
                y[idx] = t

        return FullBatchNodeSequence(self.features, self.A, self.support, self.suppG, node_indices, y, node_mask)
