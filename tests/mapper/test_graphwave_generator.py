from stellargraph.core.graph import *
from stellargraph.mapper.graphwave_generator import GraphWaveGenerator, _chebyshev, _empirical_characteristic_function

import networkx as nx
import numpy as np
import random
import pytest
import pandas as pd
import scipy.sparse as sps
import tensorflow as tf


class TestGraphWave:

    gnx = nx.barbell_graph(m1=10, m2=11)

    G = StellarGraph(gnx)
    sample_points = np.linspace(0, 100, 50).astype(np.float32)

    def test_chebyshev(self):

        generator = GraphWaveGenerator(self.G, scales=(0.1, 0.5, 1), deg=20,)

        adj = self.G.to_adjacency_matrix().tocoo()

        degree_mat = sps.diags(np.asarray(adj.sum(1)).ravel())
        laplacian = degree_mat - adj
        laplacian = np.asarray(laplacian.todense()).astype(np.float64)

        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        eigenvectors = np.asarray(eigenvectors)

        xs = np.linspace(0, eigenvalues.max(), 100)
        cheby_polys = [np.eye(laplacian.shape[0]), laplacian]

        for i in range(generator.deg - 1):
            cheby_poly = 2 * laplacian.dot(cheby_polys[-1]) - cheby_polys[-2]
            cheby_polys.append(cheby_poly)

        psi_s_list = [sum(c * x for c, x in
                          zip(
                              np.polynomial.chebyshev.chebfit(xs, np.exp(-s * xs), deg=generator.deg),
                              cheby_polys)
                          )
                      for s in generator.scales
                      ]

        dataset = tf.data.Dataset.from_tensor_slices(
            tf.sparse.eye(int(generator.laplacian.shape[0]))
        ).map(
            lambda x: _chebyshev(x, generator.laplacian, generator.coeffs, generator.deg),
            num_parallel_calls=1,
        )

        for i, x in enumerate(dataset):
            for j, psi_s in enumerate(psi_s_list):
                assert np.isclose(x.numpy()[j, :], psi_s_list[j][:, i], rtol=1e-2).all()

