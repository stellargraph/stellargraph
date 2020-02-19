from stellargraph.core.graph import *
from stellargraph.mapper.graphwave_generator import (
    GraphWaveGenerator,
    _chebyshev,
    _empirical_characteristic_function,
)

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
        """
        This test checks that the Chebyshev approximation accurately calculates the wavelets. It calculates
        the wavelets exactly using eigenvalues and compares this to the Chebyshev approximation.
        """
        scales = (1, 5, 10)

        generator = GraphWaveGenerator(self.G, scales=scales, deg=50,)

        # calculate wavelets exactly using eigenvalues
        adj = self.G.to_adjacency_matrix().tocoo()

        degree_mat = sps.diags(np.asarray(adj.sum(1)).ravel())
        laplacian = degree_mat - adj
        laplacian = np.asarray(laplacian.todense()).astype(np.float64)

        eigenvals, eigenvecs = np.linalg.eig(laplacian)
        eigenvecs = np.asarray(eigenvecs)

        psis = [
            (eigenvecs * np.exp(-s * eigenvals)).dot(eigenvecs.transpose())
            for s in scales
        ]
        psis = np.stack(psis, axis=1).astype(np.float32)
        expected_embeddings = []
        ts = tf.convert_to_tensor(self.sample_points)

        for x in tf.data.Dataset.from_tensor_slices(psis).map(
            lambda x: _empirical_characteristic_function(x, ts),
        ):
            expected_embeddings.append(x.numpy())

        expected_embeddings = np.vstack(expected_embeddings)

        actual_embeddings = [
            x.numpy()
            for x in generator.flow(
                node_ids=self.G.nodes(),
                sample_points=self.sample_points,
                batch_size=1,
                repeat=False,
            )
        ]

        actual_embeddings = np.vstack(actual_embeddings)

        # compare exactly calculated wavelets to chebyshev
        assert np.isclose(
            np.vstack(actual_embeddings), np.vstack(expected_embeddings), rtol=1e-2
        ).all()
