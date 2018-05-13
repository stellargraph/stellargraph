from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Lambda, Dropout, Reshape
from typing import List, Callable, Tuple, Dict


class MeanHinAggregator(Layer):
    """
    Mean Aggregator for HinSAGE implemented with Keras base layer

    """
    def __init__(self, output_dim: int, bias: bool = False, act: Callable = K.relu, **kwargs):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim/2)
        self.has_bias = bias
        self.act = act
        self.nr = None
        self.w_neigh = []
        self.w_self = None
        self.bias = None
        self._initializer = 'glorot_uniform'
        super(MeanHinAggregator, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix for each type of neighbour
        self.nr = len(input_shape) - 1
        self.w_neigh = [self.add_weight(
            name='w_neigh_' + str(r),
            shape=(input_shape[1+r][3], self.half_output_dim),
            initializer=self._initializer,
            trainable=True
        ) for r in range(self.nr)]

        # Weight matrix for self
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
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

        super(MeanHinAggregator, self).build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = [K.mean(z, axis=2) for z in x[1:]]

        from_self = K.dot(x[0], self.w_self)
        from_neigh = sum([K.dot(neigh_means[r], self.w_neigh[r]) for r in range(self.nr)]) / self.nr
        total = K.concatenate([from_self, from_neigh], axis=2)

        return self.act(total + self.bias if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim


class Hinsage:
    def __init__(
            self,
            output_dims: List[Dict[str, int]],
            n_samples: List[int],
            input_neigh_tree: List[Tuple[str, List[int]]],
            input_dim: Dict[str, int],
            aggregator: Layer = MeanHinAggregator,
            bias: bool = False,
            dropout: float = 0.
    ):
        """
        Construct aggregator and other supporting layers for GraphSAGE

        :param output_dims:         Output dimension at each layer
        :param n_samples:           Number of neighbours sampled for each hop/layer
        :param input_neigh_tree     Tree structure describing the neighbourhood information of the input
        :param input_dim:           Feature vector dimension
        :param aggregator:          Aggregator class
        :param bias:                Optional bias
        :param dropout:             Optional dropout
        """

        def eval_neigh_tree_per_layer(layer_info):
            reduced = [li for li in layer_info if li[1][-1] < len(layer_info)]
            return [layer_info] if len(reduced) == 0 else [layer_info] + eval_neigh_tree_per_layer(reduced)

        assert len(n_samples) == len(output_dims)
        self.n_layers = len(n_samples)
        self.n_samples = n_samples
        self.dims = [input_dim] + output_dims
        self.bias = bias
        self._dropout = Dropout(dropout)

        # Neighbourhood info per layer, and depth of each input as a result
        self.neigh_trees = eval_neigh_tree_per_layer([li for li in input_neigh_tree if len(li[1]) > 0])
        depth = [self.n_layers + 1 - sum([1 for li in [input_neigh_tree] + self.neigh_trees if i < len(li)])
                 for i in range(len(input_neigh_tree))]

        # Aggregator per node type per layer
        self._aggs = [{node_type: aggregator(output_dim,
                                             bias=self.bias,
                                             act=K.relu if layer < self.n_layers - 1 else lambda x: x)
                       for node_type, output_dim in output_dims[layer].items()}
                      for layer in range(self.n_layers)]

        # Reshape layer per neighbour per node per layer
        self._neigh_reshape = [[[Reshape((-1,
                                          self.n_samples[depth[i]],
                                          self.dims[layer][input_neigh_tree[neigh_index][0]]))
                                 for neigh_index in neigh_indices]
                                for i, (_, neigh_indices) in enumerate(self.neigh_trees[layer])]
                               for layer in range(self.n_layers)]

        self._normalization = Lambda(lambda x: K.l2_normalize(x, 2))

    def __call__(self, x: List):
        def compose_aggs(x, layer):
            """
            Function to recursively compose aggregation layers. When current layer is at final layer, then
            compose_aggs(x, layer) returns x.

            :param x:       List of feature matrix tensors
            :param layer:   Current layer index
            :return:        x computed from current layer to output layer
            """

            def neigh_list(i, neigh_indices):
                return [self._neigh_reshape[layer][i][ni](x[neigh_index])
                        for ni, neigh_index in enumerate(neigh_indices)]

            def x_next(agg: Dict[str, Layer]):
                return [agg[node_type]([self._dropout(x[i])] + [self._dropout(ne)
                                                                for ne in neigh_list(i, neigh_indices)])
                        for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[layer])]

            return compose_aggs(x_next(self._aggs[layer]), layer + 1) if layer < self.n_layers else x

        x = compose_aggs(x, 0)
        return self._normalization(x[0]) if len(x) == 1 else [self._normalization(xi) for xi in x]
