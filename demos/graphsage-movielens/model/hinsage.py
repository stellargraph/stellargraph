import tensorflow as tf
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import glorot_uniform
from typing import List, Tuple, Union, Dict, Callable


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
        assert output_dim % 2 == 0
        self.output_dim = output_dim
        self.half_dim = output_dim//2
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
            initializer=glorot_uniform(),
            trainable=True
        ) for r in range(self.nr)]

        # Weight matrix for self
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][1], self.half_dim),
            initializer=glorot_uniform(),
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
            if dropout > 0:
                out = [tf.nn.dropout(x_self, 1 - dropout)] \
                      + [tf.nn.dropout(x_neigh, 1 - dropout)
                         for x_neigh in x_neighs]
            else:
                out = [x_self] + x_neighs
            return out

        def x_next(agg_fs: Dict[str, MeanAggregatorHin]):
            return [agg_fs[schema.types[i]](apply_dropout(x[i], neigh_list(i)))
                    for i in range(schema.xlen[layer+1])]

        def create_agg_fs():
            act = K.relu if layer < schema.n_layers - 1 else lambda z: z
            return {t: agg(d[0], d[1], bias=bias, act=act)
                    for t, d in schema.dims[layer+1].items()}

        return compose_aggs(x_next(create_agg_fs()), layer + 1) if layer < schema.n_layers else x

    x = compose_aggs(x, 0)
    return tf.nn.l2_normalize(x[0], 1) if len(x) == 1 else [tf.nn.l2_normalize(xi, 1) for xi in x]

def um_bias_layer(n_nodes, ids0, ids1):
    # Weight matrix for node biases
    w_um = tf.get_variable("node_bias", shape=(n_nodes, 1),
                           initializer=tf.zeros_initializer,
                           trainable=True)

    print("embedding lookup shape:", n_nodes)

    # x0_bias and x1_bias are shape batch_size x 1
    x0_bias = tf.nn.embedding_lookup(w_um, ids0, name="x0_bias")
    x1_bias = tf.nn.embedding_lookup(w_um, ids1, name="x1_bias")
    return x0_bias, x1_bias

def hinsage_supervised(
        schema: Schema,
        batch_in: Tuple,
        agg,
        f_pred: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        f_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        bias: bool = False,
        node_bias: bool = False,
        dropout: float = 0.,
        n_nodes: int = 1,
        learning_rate: float = 0.01
    ):
    """
    Supervised HinSAGE

    :param schema:          Graph and sampling schema
    :param batch_in:        Input tensors (batch size, labels, x0, x1, ..., xn)
    :param agg:             Aggregator constructor
    :param f_pred:          Function to transform HinSAGE outputs to predictions
    :param f_loss:          Function to transform predictions and true labels to loss
    :param bias:            True for optional bias
    :param dropout:         > 0 for optional dropout
    :param learning_rate:   Learning rate for optimizer
    :return: loss, opt_op, y_preds, y_true
    """

    def _opt_op(loss, learning_rate, clip=5.0):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = [(tf.clip_by_value(grad, -clip, clip) if grad is not None else None, var)
                          for grad, var in opt.compute_gradients(loss)]
        return opt.apply_gradients(grads_and_vars)

    # Inputs
    nb, ids0, ids1, labels, *x = batch_in
    #nb = tf.Print(nb, [nb], message="Batch size: ")
    assert len(x) == schema.xlen[0]

    if node_bias:
        bias0, bias1 = um_bias_layer(n_nodes, ids0, ids1)

    else:
        bias0 = bias1 = tf.zeros((1,1))

    # HinSAGE layers
    xout = hinsage(nb, schema, agg, x, bias, dropout)

    preds = f_pred(xout, bias0, bias1)
    loss = f_loss(preds, labels)
    tf.summary.scalar('loss', loss)
    opt_op = _opt_op(loss, learning_rate)

    return loss, opt_op, preds, labels

