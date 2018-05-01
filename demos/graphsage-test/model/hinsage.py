import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from typing import List, Tuple, Union, Dict
from util.initializer import glorot_initializer


class Schema:
    """
    Schema class to create and store parameters for HinSAGE layers

    """
    def __init__(
            self,
            types: List[str],
            n_samples: List[int],
            neighs: List[List[Tuple[int, str]]],
            dims: List[Dict[str, Tuple[int, int]]],
            n_layers: int,
            xlen: List[int]
    ):
        """
        Create a graph and sampling schema. The order of elements of the list input arguments match the order in which
        the batch input feature matrices are fed through from the batch iterator. The set of feature matrices that
        compose a "batch" will be referred to as the set of sub-batches.

        :param types:       Type of nodes for each sub-batch as list of strings
        :param n_samples:   Number of samples for each sub-batch as list of integers
        :param neighs:      List of tuples (index, edge type) defining the neighbours for each sub-batch.
        :param dims:        Dict of 'node type': (feature length, number of neighbours) for each layer
        :param n_layers:    Number of hidden layers
        :param xlen:        Number of sub-batches for each layer
        """

        self.types: List[str] = types
        self.n_samples: List[int] = n_samples
        self.n_samples_cumu: List[int] = list(n_samples)
        for neigh, n_sample in zip(neighs, self.n_samples_cumu):
            for ni, nt in neigh:
                self.n_samples_cumu[ni] *= n_sample
        self.neighs: List[List[Tuple[int, str]]] = neighs
        self.dims: List[Dict[str, Tuple[int, int]]] = dims
        self.n_layers: int = n_layers
        self.xlen: List[int] = xlen


class MeanAggregatorHin(Layer):
    """
    Mean Aggregator for HinSAGE implemented with Keras base layer

    """
    def __init__(self, output_dim, nr: int, bias: bool = False, act=K.relu, **kwargs):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_dim = int(output_dim/2)
        self.nr = nr  # Number of neighbour edge types
        self.w_neigh = []
        self.has_bias = bias
        self.act = act
        super(MeanAggregatorHin, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix for each type of neighbour
        self.w_neigh = [self.add_weight(
            name='w_neigh_' + str(r),
            shape=(input_shape[1+r][2], self.half_dim),
            initializer=glorot_initializer((input_shape[1+r][2], int(self.half_dim))),
            trainable=True
        ) for r in range(self.nr)]

        # Weight matrix for self
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][1], self.half_dim),
            initializer=glorot_initializer((input_shape[0][1], int(self.half_dim))),
            trainable=True
        )

        # Optional bias
        if self.has_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.output_dim],
                initializer='zeros',
                trainable=True
            )

        super(MeanAggregatorHin, self).build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = [K.mean(z, axis=1) for z in x[1:]]

        from_self = K.dot(x[0], self.w_self)
        from_neigh = sum([K.dot(neigh_means[r], self.w_neigh[r]) for r in range(self.nr)]) / self.nr
        total = K.concatenate([from_self, from_neigh], axis=1)

        return self.act(total + self.bias if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.output_dim


def hinsage(
        nb: Union[tf.Tensor, int],
        schema: Schema,
        agg,
        x: List[tf.Tensor],
        bias: bool,
        dropout: float
):
    """
    HinSAGE layers with given aggregator

    :param nb:      Batch size
    :param schema:  Graph and sampling schema
    :param agg:     Aggregator constructor
    :param x:       Feature matrices for each hop
    :param bias:    True for optional bias
    :param dropout: > 0 for optional dropout
    :return: Node embeddings for given batch
    """

    def compose_aggs(x, layer):
        """
        Function to recursively compose aggregation layers. When current layer is at final layer, then
        compose_aggs(x, layer) returns x.

        :param x:       List of feature matrix tensors
        :param layer:   Current layer index
        :return:        x computed from current layer to output layer
        """

        def neigh_reshape(ni, i):
            return tf.reshape(
                x[ni],
                [
                    nb * schema.n_samples_cumu[i],
                    schema.n_samples[ni],
                    schema.dims[layer][schema.types[ni]][0]
                ]
            )

        def neigh_list(i):
            return [neigh_reshape(ni[0], i) for ni in schema.neighs[i]]

        def apply_dropout(x_self, x_neighs):
            return [tf.nn.dropout(x_self, 1 - dropout)] + [tf.nn.dropout(x_neigh, 1 - dropout) for x_neigh in x_neighs]

        def x_next(agg_fs: Dict[str, MeanAggregatorHin]):
            return [agg_fs[schema.types[i]](apply_dropout(x[i], neigh_list(i)))
                    for i in range(schema.xlen[layer+1])]

        def create_agg_fs():
            return {t: agg(d[0], d[1], bias=bias, act=tf.nn.relu if layer < schema.n_layers - 1 else lambda z: z)
                    for t, d in schema.dims[layer+1].items()}

        return compose_aggs(x_next(create_agg_fs()), layer + 1) if layer < schema.n_layers else x

    x = compose_aggs(x, 0)
    return tf.nn.l2_normalize(x[0], 1) if len(x) == 1 else [tf.nn.l2_normalize(xi, 1) for xi in x]


def hinsage_nai(
        num_labels: int,
        schema: Schema,
        batch_in: Tuple,
        agg,
        sigmoid: bool = False,
        bias: bool = False,
        dropout: float = 0.,
        learning_rate: float = 0.01
):
    """
    Node Attribute Inference with HinSAGE

    :param num_labels:      Total number of possible labels
    :param schema:          Graph and sampling schema
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
    assert len(x) == schema.xlen[0]

    # hinsage layers
    x_out = hinsage(nb, schema, agg, x, bias, dropout)

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


def hinsage_lai(
        num_labels: int,
        schema: Schema,
        batch_in: Tuple,
        agg,
        sigmoid: bool = False,
        bias: bool = False,
        dropout: float = 0.,
        learning_rate: float = 0.01
):
    """
    Link Attribute Inference with HinSAGE

    :param num_labels:      Total number of possible labels
    :param schema:          Graph and sampling schema
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

    nb, labels, *x = batch_in
    nb = tf.Print(nb, [nb], message="Batch size: ")
    assert len(x) == schema.xlen[0]

    # hinsage layers
    x_out = hinsage(nb, schema, agg, x, bias, dropout)

    # concatenate output to form edge representations
    x_edge = tf.nn.l2_normalize(tf.concat(x_out, axis=1))

    # loss
    preds = Dense(num_labels)(x_edge)
    loss = _loss(preds, labels, sigmoid)
    tf.summary.scalar('loss', loss)

    # optimizer
    opt_op = _opt_op(loss, learning_rate)

    # predictions
    y_preds = _pred(preds, sigmoid)
    y_true = labels

    return loss, opt_op, y_preds, y_true


