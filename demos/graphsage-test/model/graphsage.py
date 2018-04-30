import tensorflow as tf
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from typing import List, Tuple, Union
from util.initializer import glorot_initializer


class MeanAggregator(Layer):
    """
    Mean Aggregator for GraphSAGE implemented with Keras base layer

    """

    def __init__(self, output_dim, bias=False, act=K.relu, **kwargs):
        self.output_dim = output_dim
        self.has_bias = bias
        self.act = act
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
        if self.has_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[2*self.output_dim],
                initializer='zeros',
                trainable=True
            )
        super(MeanAggregator, self).build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = K.mean(x[1], axis=1)

        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_means, self.w_neigh)
        total = K.concatenate([from_self, from_neigh], axis=1)

        return self.act(total + self.bias if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 2*self.output_dim


def graphsage(
        nb: Union[tf.Tensor, int],
        ns: List[int],
        dims: List[int],
        agg,
        x: List[tf.Tensor],
        bias: bool,
        dropout: float
):
    """
    GraphSAGE layers with given aggregator

    :param nb:      Batch size
    :param ns:      Number of neighbours sampled at each hop/layer
    :param dims:    Length of feature vector at each layer
    :param agg:     Aggregator constructor
    :param x:       Feature matrices for each hop
    :param bias:    True for optional bias
    :param dropout: > 0 for optional dropout
    :return: Node embeddings for given batch
    """

    nl = len(ns)
    ns += [1]
    output_dims = dims[1:]
    input_dims = dims[0:1] + [2*d for d in dims[1:-1]]  # input dims are doubled due to concatenation

    # function to recursively compose aggregators at layer
    def compose_aggs(x, layer):
        def neigh_reshape(xi, i):
            return tf.reshape(xi, [nb*ns[-i-1], ns[-i-2], input_dims[layer]])

        def apply_dropout(x_self, x_neigh):
            return [tf.nn.dropout(x_self, 1 - dropout), tf.nn.dropout(x_neigh, 1 - dropout)]

        def x_next(agg_f, x):
            return [agg_f(apply_dropout(x[i], neigh_reshape(x[i+1], i))) for i in range(nl - layer)]

        def create_agg_f():
            return agg(output_dims[layer], bias=bias, act=tf.nn.relu if layer < nl - 1 else lambda z: z)

        return compose_aggs(x_next(create_agg_f(), x), layer + 1) if layer < nl else x[0]

    return tf.nn.l2_normalize(compose_aggs(x, 0), 1)


def graphsage_nai(
        num_labels: int,
        dims: List[int],
        num_samples: List[int],
        batch_in: Tuple,
        agg,
        sigmoid: bool = False,
        bias: bool = False,
        dropout: float = 0.,
        learning_rate: float = 0.01
):
    """
    Node Attribute Inference with GraphSAGE

    :param num_labels:      Total number of possible labels
    :param dims:            Feature vector lengths for each layer
    :param num_samples:     Number of neighbours sampled for each hop/layer
    :param batch_in:        Input tensors (batch size, labels, x0, x1, ..., xn)
    :param agg:             Aggregator constructor
    :param sigmoid:         True for multiple true labels for each node
    :param bias:            True for optional bias
    :param dropout:         > 0 for optional dropout
    :param learning_rate:   Learning rate for optimizer
    :return: loss, opt_op, y_preds, y_true
    """

    def _loss(preds, labels, sigmoid):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels) if sigmoid
            else tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        )

    def _pred(preds, sigmoid):
        return tf.nn.sigmoid(preds) if sigmoid else tf.nn.softmax(preds)

    def _opt_op(loss, learning_rate):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                          for grad, var in opt.compute_gradients(loss)]
        return opt.apply_gradients(grads_and_vars)

    # check inputs
    nb, labels, *x = batch_in
    nb = tf.Print(nb, [nb], message="Batch size: ")
    assert len(x) == len(num_samples) + 1 and len(x) == len(dims)

    # graphsage layers
    x_out = graphsage(nb, num_samples, dims, agg, x, bias, dropout)

    # loss
    preds = Dense(num_labels)(x_out)
    loss = _loss(preds, labels, sigmoid)
    tf.summary.scalar('loss', loss)

    # optimizer
    opt_op = _opt_op(loss, learning_rate)

    # predictions
    y_preds = _pred(preds, sigmoid)
    y_true = labels

    return loss, opt_op, y_preds, y_true


def graphsage_lai(
        num_labels: int,
        dims: List[int],
        num_samples: List[int],
        batch_in: Tuple,
        agg,
        sigmoid: bool = False,
        bias: bool = False,
        dropout: float = 0.,
        learning_rate: float = 0.01
):
    """
    Node Attribute Inference with GraphSAGE

    :param num_labels:      Total number of possible labels
    :param dims:            Feature vector lengths for each layer
    :param num_samples:     Number of neighbours sampled for each hop/layer
    :param batch_in:        Input tensors (batch size, labels, xs0, xs1, ..., xsn, xd0, xd1, ..., xdn)
    :param agg:             Aggregator constructor
    :param sigmoid:         True for multiple true labels for each node
    :param bias:            True for optional bias
    :param dropout:         > 0 for optional dropout
    :param learning_rate:   Learning rate for optimizer
    :return: loss, opt_op, y_preds, y_true
    """

    nb, labels, *x = batch_in
    x1 = x[:len(x)/2]
    x2 = x[len(x)/2:]
    assert len(x1) == len(x2) and len(x1) == len(num_samples) + 1 and len(x1) == len(dims)
    raise NotImplementedError

