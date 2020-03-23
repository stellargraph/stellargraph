from stellargraph.losses import graph_log_likelihood
import numpy as np
import tensorflow as tf


def test_graph_log_likelihood():

    batch_rows = 7

    batch_adj = (np.random.random((batch_rows, 1, 100)) > 0.7).astype(np.float32)

    expected_walks = np.random.random((batch_rows, 1, 100)).astype(np.float32)
    scores = np.random.random((batch_rows, 1, 100)).astype(np.float32)

    wys_output = np.concatenate((expected_walks, scores), axis=1)

    actual_loss = graph_log_likelihood(batch_adj, wys_output).numpy()[0]

    sigmoid_scores = 1 / (1 + np.exp(-scores))
    expected_loss = np.abs(
        -expected_walks * np.log(sigmoid_scores)
        - (batch_adj == 0) * np.log(1 - sigmoid_scores)
    )

    expected_loss = expected_loss.sum()

    assert np.allclose(actual_loss, expected_loss, rtol=0.01)
