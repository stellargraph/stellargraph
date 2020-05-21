from stellargraph.layer import APPNP, GAT
from stellargraph.mapper import ClusterNodeGenerator
import tensorflow as tf
import numpy as np
from ..test_utils.graphs import example_graph_random
import pytest


@pytest.mark.parametrize("model_type", [APPNP, GAT])
def test_fullbatch_cluster_models(model_type):
    G = example_graph_random(n_nodes=50)
    generator = ClusterNodeGenerator(G, clusters=10)
    nodes = G.nodes()[:40]
    gen = generator.flow(nodes, targets=np.ones(len(nodes)))

    gnn = model_type(
        generator=generator,
        layer_sizes=[16, 16, 1],
        activations=["relu", "relu", "relu"],
    )

    model = tf.keras.Model(*gnn.in_out_tensors())
    model.compile(optimizer="adam", loss="binary_crossentropy")
    history = model.fit(gen, validation_data=gen, epochs=2)
    results = model.evaluate(gen)

    # this doesn't work for any cluster models including ClusterGCN
    # because the model spits out predictions with shapes:
    # [(1, cluster_1_size, feat_size), (1, cluster_2_size, feat_size)...]
    # and attempts to concatenate along axis 0
    # predictions = model.predict(gen)
    x_in, x_out = gnn.in_out_tensors()
    x_out_flat = tf.squeeze(x_out, 0)
    embedding_model = tf.keras.Model(inputs=x_in, outputs=x_out_flat)
    predictions = embedding_model.predict(gen)

    assert predictions.shape == (len(nodes), 1)
