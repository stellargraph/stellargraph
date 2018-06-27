# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Heterogeneous GraphSAGE and compatible aggregator layers

"""

from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Lambda, Dropout, Reshape, Activation
from typing import List, Callable, Tuple, Dict, Union, AnyStr


class MeanHinAggregator(Layer):
    """
    Mean Aggregator for HinSAGE implemented with Keras base layer

    """

    def __init__(
        self, output_dim: int, bias: bool = False, act: Callable = K.relu, **kwargs
    ):
        self.output_dim = output_dim
        assert output_dim % 2 == 0
        self.half_output_dim = int(output_dim / 2)
        self.has_bias = bias
        self.act = act
        self.nr = None
        self.w_neigh = []
        self.w_self = None
        self.bias = None
        self._initializer = "glorot_uniform"
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix for each type of neighbour
        self.nr = len(input_shape) - 1
        self.w_neigh = [
            self.add_weight(
                name="w_neigh_" + str(r),
                shape=(input_shape[1 + r][3], self.half_output_dim),
                initializer=self._initializer,
                trainable=True,
            )
            for r in range(self.nr)
        ]

        # Weight matrix for self
        self.w_self = self.add_weight(
            name="w_self",
            shape=(input_shape[0][2], self.half_output_dim),
            initializer=self._initializer,
            trainable=True,
        )

        # Optional bias
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, x, **kwargs):
        neigh_means = [K.mean(z, axis=2) for z in x[1:]]

        from_self = K.dot(x[0], self.w_self)
        from_neigh = (
            sum([K.dot(neigh_means[r], self.w_neigh[r]) for r in range(self.nr)])
            / self.nr
        )
        total = K.concatenate([from_self, from_neigh], axis=2)   #YT: this corresponds to concat=Partial
        # TODO: implement concat=Full and concat=False
        actx = self.act(total + self.bias if self.has_bias else total)

        return Activation(self.act, name=kwargs.get("name"))(actx)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim


class Hinsage:
    """
    Implementation of the GraphSAGE algorithm extended for heterogeneous graphs with Keras layers.
    """

    def __init__(
        self,
        output_dims: List[Union[Dict[str, int], int]],
        n_samples: List[int],
        input_neigh_tree: List[Tuple[str, List[int]]],
        input_dim: Dict[str, int],
        aggregator: Layer = MeanHinAggregator,
        bias: bool = False,
        dropout: float = 0.,
    ):
        """
        Construct aggregator and other supporting layers for HinSAGE

        :param output_dims:         Output dimension at each layer
        :param n_samples:           Number of neighbours sampled for each hop/layer
        :param input_neigh_tree     Tree structure describing the neighbourhood information of the input
        :param input_dim:           Feature vector dimension
        :param aggregator:          Aggregator class
        :param bias:                Optional bias
        :param dropout:             Optional dropout
        """

        def eval_neigh_tree_per_layer(input_tree):
            """
            Function to evaluate the neighbourhood tree structure for every layer

            :param input_tree:  Neighbourhood tree for the input batch
            :return:            List of neighbourhood trees
            """

            reduced = [li for li in input_tree if li[1][-1] < len(input_tree)]
            return (
                [input_tree]
                if len(reduced) == 0
                else [input_tree] + eval_neigh_tree_per_layer(reduced)
            )

        assert len(n_samples) == len(output_dims)
        self.n_layers = len(n_samples)
        self.n_samples = n_samples
        self.bias = bias
        self._dropout = Dropout(dropout)

        # Neighbourhood info per layer
        self.neigh_trees = eval_neigh_tree_per_layer(
            [li for li in input_neigh_tree if len(li[1]) > 0]
        )

        # Depth of each input i.e. number of hops from root nodes
        depth = [
            self.n_layers
            + 1
            - sum([1 for li in [input_neigh_tree] + self.neigh_trees if i < len(li)])
            for i in range(len(input_neigh_tree))
        ]

        # Dict of {node type: dimension} per layer
        self.dims = [
            dim
            if isinstance(dim, dict)
            else {k: dim for k, _ in ([input_neigh_tree] + self.neigh_trees)[layer]}
            for layer, dim in enumerate([input_dim] + output_dims)
        ]

        # Dict of {node type: aggregator} per layer
        self._aggs = [
            {
                node_type: aggregator(
                    output_dim,
                    bias=self.bias,
                    act=K.relu if layer < self.n_layers - 1 else lambda x: x,
                )
                for node_type, output_dim in self.dims[layer + 1].items()
            }
            for layer in range(self.n_layers)
        ]

        # Reshape object per neighbour per node per layer
        self._neigh_reshape = [
            [
                [
                    Reshape(
                        (
                            -1,
                            self.n_samples[depth[i]],
                            self.dims[layer][input_neigh_tree[neigh_index][0]],
                        )
                    )
                    for neigh_index in neigh_indices
                ]
                for i, (_, neigh_indices) in enumerate(self.neigh_trees[layer])
            ]
            for layer in range(self.n_layers)
        ]

        self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=2))

    def __call__(self, x: List):
        """
        Apply aggregator layers

        :param x:       Batch input features
        :return:        Output tensor
        """

        def compose_layers(x: List, layer: int):
            """
            Function to recursively compose aggregation layers. When current layer is at final layer, then
            compose_layers(x, layer) returns x.

            :param x:       List of feature matrix tensors
            :param layer:   Current layer index
            :return:        x computed from current layer to output layer
            """

            def neigh_list(i, neigh_indices):
                return [
                    self._neigh_reshape[layer][i][ni](x[neigh_index])
                    for ni, neigh_index in enumerate(neigh_indices)
                ]

            def x_next(agg: Dict[str, Layer]):
                return [
                    agg[node_type](
                        [
                            self._dropout(x[i]),
                            *[self._dropout(ne) for ne in neigh_list(i, neigh_indices)],
                        ],
                        name="{}_{}".format(node_type, layer),
                    )
                    for i, (node_type, neigh_indices) in enumerate(
                        self.neigh_trees[layer]
                    )
                ]

            return (
                compose_layers(x_next(self._aggs[layer]), layer + 1)
                if layer < self.n_layers
                else x
            )

        x = compose_layers(x, 0)
        return (
            self._normalization(x[0])
            if len(x) == 1
            else [self._normalization(xi) for xi in x]
        )
