from .fixtures import assert_reproducible
from stellargraph.layer import DeepGraphInfomax, GCN, APPNP, GAT, PPNP
from stellargraph.mapper import FullBatchNodeGenerator, CorruptedGenerator
from ..test_utils.graphs import example_graph_random
import tensorflow as tf
import pytest
import numpy as np


def dgi(generator, gen, model_type):

    tf.random.set_seed(1234)
    np.random.seed(1234)

    emb_dim = 4

    base_model = model_type(
        generator=generator, activations=["relu"], layer_sizes=[emb_dim]
    )
    infomax = DeepGraphInfomax(base_model)

    model = tf.keras.Model(*infomax.build())
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer="Adam")
    model.fit(gen)
    return model


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT])
def test_dgi_sparse(model_type):
    G = example_graph_random()
    generator = FullBatchNodeGenerator(G, sparse=True)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    assert_reproducible(lambda: dgi(generator, gen, model_type), num_iter=3)


@pytest.mark.parametrize("model_type", [GCN, APPNP, GAT, PPNP])
def test_dgi_dense(model_type):
    G = example_graph_random()
    generator = FullBatchNodeGenerator(G, sparse=True)
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(G.nodes())

    assert_reproducible(lambda: dgi(generator, gen, model_type), num_iter=3)
