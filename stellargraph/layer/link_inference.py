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
Link inference functions for link classification (including link prediction) and
link attribute inference (regression)
"""

from typing import AnyStr, Optional, List, Tuple
from keras.layers import Layer, Concatenate, Dense, Lambda, Multiply, Average, Reshape
from keras import backend as K


class LeakyClippedLinear(Layer):
    """
    Leaky Clipped Linear Unit.

        Args:
            low (float): Lower threshold
            high (float): Lower threshold
            alpha (float) The slope of the function below low or above high.
    """

    def __init__(
        self, low: float = 1.0, high: float = 5.0, alpha: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.gamma = K.cast_to_floatx(1 - alpha)
        self.lo = K.cast_to_floatx(low)
        self.hi = K.cast_to_floatx(high)

    def call(self, x, mask=None):
        x_lo = K.relu(self.lo - x)
        x_hi = K.relu(x - self.hi)
        return x + self.gamma * x_lo - self.gamma * x_hi

    def get_config(self):
        config = {
            "alpha": float(1 - self.gamma),
            "low": float(self.lo),
            "high": float(self.hi),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def link_inference(
    output_dim: int = 1,
    output_act: AnyStr = "linear",
    edge_feature_method: AnyStr = "ip",
    clip_limits: Optional[Tuple[float]] = None,
    name: AnyStr = "link_inference",
):
    """
    Defines an edge inference function that takes source, destination node features as input,
    and returns a numeric vector of output_dim size.

    Args:
        output_dim (int): Number of predictor's output units -- desired dimensionality of the output.
        output_act (str), optional: activation function applied to the output, one of "softmax", "sigmoid", etc.,
            or any activation function supported by Keras, see https://keras.io/activations/ for more information.
        edge_feature_method (str), optional: Name of the method of combining (src,dst) node features into edge features.
            One of
             * 'concat' -- concatenation,
             * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
             * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
             * 'l1' -- :math:`l_1(u,v)_i = |u_i-v_i|`,
             * 'l2' -- :math:`l_2(u,v)_i = (u_i-v_i)^2`,
             * 'avg' -- :math:`avg(u,v) = (u+v)/2`.
        clip_limits (Tuple[float]): lower and upper thresholds for LeakyClippedLinear unit on top. If None (not provided),
            the LeakyClippedLinear unit is not applied.
        name (str): optional name of the defined function, used for error logging

    Returns:
        Function taking edge tensors with src, dst node features (i.e., pairs of (node_src, node_dst) tensors) and
        returning a vector of output_dim length (e.g., edge class probabilities, edge attribute prediction, etc.).
    """

    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if edge_feature_method == "ip" or edge_feature_method == "dot":
            out = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                [x0, x1]
            )

        elif edge_feature_method == "l1":
            # l1(u,v)_i = |u_i - v_i| - vector of the same size as u,v
            le = Lambda(lambda x: K.abs(x[0] - x[1]))([x0, x1])
            # add dense layer to convert le to the desired output:
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "l2":
            # l2(u,v)_i = (u_i - v_i)^2 - vector of the same size as u,v
            le = Lambda(lambda x: K.square(x[0] - x[1]))([x0, x1])
            # add dense layer to convert le to the desired output:
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "mul" or edge_feature_method == "hadamard":
            le = Multiply()([x0, x1])
            # add dense layer to convert le to the desired output:
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "concat":
            le = Concatenate()([x0, x1])
            # add dense layer to convert le to the desired output:
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "avg":
            le = Average()([x0, x1])
            # add dense layer to convert le to the desired output:
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        else:
            raise NotImplementedError(
                "{}: the requested method '{}' is not known/not implemented".format(
                    name, edge_feature_method
                )
            )

        if clip_limits:
            out = LeakyClippedLinear(
                low=clip_limits[0], high=clip_limits[1], alpha=0.1
            )(out)

        return out

    print(
        "{}: using '{}' method to combine node embeddings into edge embeddings".format(
            name, edge_feature_method
        )
    )
    return edge_function


def link_classification(
    output_dim: int = 1,
    output_act: AnyStr = "sigmoid",
    edge_feature_method: AnyStr = "ip",
):
    """
    Defines a function that predicts a binary or multi-class edge classification output from
    (source, destination) node features.

    Args:
        output_dim (int): Number of classifier's output units -- desired dimensionality of the output,
        output_act (str), optional: activation function applied to the output, one of "softmax", "sigmoid", etc.,
            or any activation function supported by Keras, see https://keras.io/activations/ for more information.
        edge_feature_method (str), optional: Name of the method of combining (src,dst) node features into edge features.
            One of:
             * 'concat' -- concatenation,
             * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
             * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
             * 'l1' -- :math:`l_1(u,v)_i = |u_i-v_i|`,
             * 'l2' -- :math:`l_2(u,v)_i = (u_i-v_i)^2`,
             * 'avg' -- :math:`avg(u,v) = (u+v)/2`.

    Returns:
        Function taking edge tensors with src, dst node features (i.e., pairs of (node_src, node_dst) tensors) and
        returning logits of output_dim length (e.g., edge class probabilities).
    """

    edge_function = link_inference(
        output_dim=output_dim,
        output_act=output_act,
        edge_feature_method=edge_feature_method,
        name="link_classification",
    )

    return edge_function


def link_regression(
    output_dim: int = 1,
    clip_limits: Optional[Tuple[float]] = None,
    edge_feature_method: AnyStr = "ip",
):
    """
    Defines a function that predicts a numeric edge regression output vector/scalar from
    (source, destination) node features.

    Args:
        output_dim (int): Number of classifier's output units -- desired dimensionality of the output,
        clip_limits (tuple): lower and upper thresholds for LeakyClippedLinear unit on top. If None (not provided),
            the LeakyClippedLinear unit is not applied.
        edge_feature_method (str), optional: Name of the method of combining (src,dst) node features into edge features.
            One of:
             * 'concat' -- concatenation,
             * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
             * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
             * 'l1' -- :math:`l_1(u,v)_i = |u_i-v_i|`,
             * 'l2' -- :math:`l_2(u,v)_i = (u_i-v_i)^2`,
             * 'avg' -- :math:`avg(u,v) = (u+v)/2`.

    Returns:
        Function taking edge tensors with src, dst node features (i.e., pairs of (node_src, node_dst) tensors) and
        returning a numeric value (e.g., edge attribute being predicted) constructed according to edge_feature_method.
    """

    edge_function = link_inference(
        output_dim=output_dim,
        output_act="linear",
        edge_feature_method=edge_feature_method,
        clip_limits=clip_limits,
        name="link_regression",
    )

    return edge_function
