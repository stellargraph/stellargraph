import tensorflow as tf


def graph_log_likelihood(y_true, y_pred):
    batch_adj = tf.gather(y_true, [0], axis=1)
    expected_walks = tf.gather(y_pred, [0], axis=1)
    sigmoids = tf.gather(y_pred, [1], axis=1)
    adj_mask = tf.cast((batch_adj == 0), 'float32')

    loss = tf.math.reduce_mean(
        tf.abs(
            -(expected_walks * tf.log(sigmoids + 1e-6)) - adj_mask * tf.log(1 - sigmoids + 1e-6)
        )
    )

    return tf.expand_dims(loss, 0)
