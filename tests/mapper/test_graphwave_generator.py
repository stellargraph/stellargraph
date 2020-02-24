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

from stellargraph.core.graph import *
from stellargraph.mapper.graphwave_generator import (
    GraphWaveGenerator,
    _empirical_characteristic_function,
)

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sps
import tensorflow as tf


class TestGraphWave:

    gnx = nx.barbell_graph(m1=10, m2=11)

    G = StellarGraph(gnx)
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    num_nodes = len(G.nodes())

    def test_init(self):
        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        assert np.isclose(generator.scales, (0.1, 2, 3, 4)).all()
        assert generator.degree == 10
        assert generator.coeffs.shape == (4, 10 + 1)
        assert generator.laplacian.shape == (len(self.gnx.nodes), len(self.gnx.nodes))

    def test_bad_init(self):

        with pytest.raises(TypeError):
            generator = GraphWaveGenerator(None, scales=(0.1, 2, 3, 4), degree=10)

        with pytest.raises(TypeError, match="deg: expected.*found float"):
            generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=1.1)

        with pytest.raises(ValueError, match="deg: expected.*found 0"):
            generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=0)

    def test_bad_flow(self):
        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        with pytest.raises(TypeError, match="batch_size: expected.*found float"):
            generator.flow(self.G.nodes(), self.sample_points, batch_size=4.5)

        with pytest.raises(ValueError, match="batch_size: expected.*found 0"):
            generator.flow(self.G.nodes(), self.sample_points, batch_size=0)

        with pytest.raises(TypeError, match="shuffle: expected.*found int"):
            generator.flow(self.G.nodes(), self.sample_points, batch_size=1, shuffle=1)

        with pytest.raises(TypeError, match="repeat: expected.*found int"):
            generator.flow(self.G.nodes(), self.sample_points, batch_size=1, repeat=1)

        with pytest.raises(
            TypeError, match="num_parallel_calls: expected.*found float"
        ):
            generator.flow(
                self.G.nodes(), self.sample_points, batch_size=1, num_parallel_calls=2.2
            )

        with pytest.raises(ValueError, match="num_parallel_calls: expected.*found 0"):
            generator.flow(
                self.G.nodes(), self.sample_points, batch_size=1, num_parallel_calls=0
            )

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_flow_shuffle(self, shuffle):

        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        embeddings_dataset = generator.flow(
            node_ids=self.G.nodes(),
            sample_points=self.sample_points,
            batch_size=1,
            repeat=False,
            shuffle=shuffle,
        )

        first, *rest = [
            np.vstack([x.numpy() for x in embeddings_dataset]) for _ in range(20)
        ]

        if shuffle:
            assert any((first == r).all() for r in rest) is False
        else:
            assert all((first == r).all() for r in rest) is True

    def test_determinism(self):

        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        embeddings_dataset = generator.flow(
            node_ids=self.G.nodes(),
            sample_points=self.sample_points,
            batch_size=1,
            repeat=False,
            shuffle=True,
            seed=1234,
        )

        first = np.vstack([x.numpy() for x in embeddings_dataset])

        embeddings_dataset = generator.flow(
            node_ids=self.G.nodes(),
            sample_points=self.sample_points,
            batch_size=1,
            repeat=False,
            shuffle=True,
            seed=1234,
        )

        second = np.vstack([x.numpy() for x in embeddings_dataset])

        assert (first == second).all()

    @pytest.mark.parametrize("repeat", [False, True])
    def test_flow_repeat(self, repeat):
        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        for i, x in enumerate(
            generator.flow(
                self.G.nodes(),
                sample_points=self.sample_points,
                batch_size=1,
                repeat=repeat,
            )
        ):

            if i > self.num_nodes:
                break

        assert (i > self.num_nodes) == repeat

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_flow_batch_size(self, batch_size):

        scales = (0.1, 2, 3, 4)
        generator = GraphWaveGenerator(self.G, scales=scales, degree=10)

        expected_embed_dim = len(self.sample_points) * len(scales) * 2

        for i, x in enumerate(
            generator.flow(
                self.G.nodes(),
                sample_points=self.sample_points,
                batch_size=batch_size,
                repeat=False,
            )
        ):
            # all batches except maybe last will have a batch size of batch_size
            if i < self.num_nodes // batch_size:
                assert x.shape == (batch_size, expected_embed_dim)
            else:
                assert x.shape == (self.num_nodes % batch_size, expected_embed_dim)

    @pytest.mark.parametrize("num_samples", [1, 25, 50])
    def test_embedding_dim(self, num_samples):
        scales = (0.1, 2, 3, 4)
        generator = GraphWaveGenerator(self.G, scales=scales, degree=10)

        sample_points = np.linspace(0, 1, num_samples)

        expected_embed_dim = len(sample_points) * len(scales) * 2

        for x in generator.flow(
            self.G.nodes(), sample_points=sample_points, batch_size=4, repeat=False
        ):

            assert x.shape[1] == expected_embed_dim

    def test_flow_targets(self):

        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        for i, x in enumerate(
            generator.flow(
                self.G.nodes(),
                sample_points=self.sample_points,
                batch_size=1,
                targets=np.arange(self.num_nodes),
            )
        ):
            assert len(x) == 2
            assert x[1].numpy() == i

    def test_flow_node_ids(self):

        generator = GraphWaveGenerator(self.G, scales=(0.1, 2, 3, 4), degree=10)

        node_ids = list(self.G.nodes())[:4]

        expected_targets = generator._node_lookup(node_ids)
        actual_targets = []
        for x in generator.flow(
            node_ids,
            sample_points=self.sample_points,
            batch_size=1,
            targets=expected_targets,
        ):

            actual_targets.append(x[1].numpy())

        assert all(a == b for a, b in zip(expected_targets, actual_targets))

    def test_chebyshev(self):
        """
        This test checks that the Chebyshev approximation accurately calculates the wavelets. It calculates
        the wavelets exactly using eigenvalues and compares this to the Chebyshev approximation.
        """
        scales = (1, 5, 10)

        generator = GraphWaveGenerator(self.G, scales=scales, degree=50,)

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
