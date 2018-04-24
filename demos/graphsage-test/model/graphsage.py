import tensorflow as tf
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from typing import List, Tuple
from util.initializer import glorot_initializer


class MeanAggregator(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MeanAggregator, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w_neigh = self.add_weight(
            name='w_neigh',
            shape=(input_shape[1][2], self.output_dim),
            initializer=glorot_initializer((input_shape[1][2], self.output_dim)),
            trainable=True
        )
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][1], self.output_dim),
            initializer=glorot_initializer((input_shape[0][1], self.output_dim)),
            trainable=True
        )
        super(MeanAggregator, self).build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = K.mean(x[1], axis=1)

        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_means, self.w_neigh)
        return K.concatenate([from_self, from_neigh], axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 2*self.output_dim


def graphsage(nb, ns, dims, agg, x):
    nl = len(ns)
    ns += [1]
    output_dims = dims[1:]
    input_dims = dims[0:1] + [2*d for d in dims[1:-1]]  # input dims are doubled due to concatenation

    # function to recursively compose aggregators at layer
    def compose_aggs(x, layer):
        def neigh_reshape(xi, i):
            return tf.reshape(xi, [nb*ns[-i-1], ns[-i-2], input_dims[layer]])

        def x_next(agg_f, x):
            return [agg_f([x[i], neigh_reshape(x[i+1], i)]) for i in range(nl - layer)]

        return compose_aggs(x_next(agg(output_dims[layer]), x), layer + 1) if layer < nl else x[0]

    return tf.nn.l2_normalize(compose_aggs(x, 0), 1)


def supervised_graphsage(
        num_labels: int,
        dims: List[int],
        num_samples: List[int],
        batch_in: Tuple,
        agg
):
    # check inputs
    nb, labels, *x = batch_in
    nb = tf.Print(nb, [nb])
    assert len(x) == len(num_samples) + 1 and len(x) == len(dims)

    # graphsage
    x_out = graphsage(nb, num_samples, dims, agg, x)

    # loss
    preds = Dense(num_labels)(x_out)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                              for grad, var in grads_and_vars]
    opt_op = optimizer.apply_gradients(clipped_grads_and_vars)

    # predictions
    y_preds = tf.nn.sigmoid(preds)
    y_true = labels

    return loss, opt_op, y_preds, y_true


