import tensorflow as tf

from ..core.experimental import experimental


@experimental(reason="lack of unit tests")
def graph_log_likelihood(y_true, y_pred):
    """
    .. warning::

        This function is experimental: it is insufficiently tested.
    """
    batch_adj = tf.gather(y_true, [0], axis=1)

    expected_walks = tf.gather(y_pred, [0], axis=1)
    sigmoids = tf.gather(y_pred, [1], axis=1)

    adj_mask = tf.cast((batch_adj == 0), "float32")

    loss = tf.math.reduce_sum(
        tf.abs(
            -(expected_walks * tf.math.log(sigmoids + 1e-9))
            - adj_mask * tf.math.log(1 - sigmoids + 1e-9)
        )
    )

    return tf.expand_dims(loss, 0)
