# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Concatenate,
    Dense,
    Lambda,
    Multiply,
    Average,
    Reshape,
    Activation,
)
from tensorflow.keras import backend as K
import warnings


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


class LinkEmbedding(Layer):
    """
    Defines an edge inference function that takes source, destination node embeddings
    (node features) as input, and returns a numeric vector of output_dim size.

    This class takes as input as either:

     * A list of two tensors of shape (N, M) being the embeddings for each of the nodes in the link,
       where N is the number of links, and M is the node embedding size.
     * A single tensor of shape (..., N, 2, M) where the axis second from last indexes the nodes
       in the link and N is the number of links and M the embedding size.

    Examples:
        Consider two tensors containing the source and destination embeddings of size M::

            x_src = tf.constant(x_src, shape=(1, M), dtype="float32")
            x_dst = tf.constant(x_dst, shape=(1, M), dtype="float32")

            li = LinkEmbedding(method="ip", activation="sigmoid")([x_src, x_dst])

    Args:
        axis (int): If a single tensor is supplied this is the axis that indexes the node
            embeddings so that the indices 0 and 1 give the node embeddings to be combined.
            This is ignored if two tensors are supplied as a list.
        activation (str), optional: activation function applied to the output, one of "softmax", "sigmoid", etc.,
            or any activation function supported by Keras, see https://keras.io/activations/ for more information.
        method (str), optional: Name of the method of combining (src,dst) node features or embeddings into edge embeddings.
            One of:
            * 'concat' -- concatenation,
            * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
            * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
            * 'l1' -- L1 operator, :math:`l_1(u,v)_i = |u_i-v_i|`,
            * 'l2' -- L2 operator, :math:`l_2(u,v)_i = (u_i-v_i)^2`,
            * 'avg' -- average, :math:`avg(u,v) = (u+v)/2`.
            For all methods except 'ip' or 'dot' a dense layer is applied on top of the combined
            edge embedding to transform to a vector of size `output_dim`.

    """

    def __init__(
        self,
        method: AnyStr = "ip",
        axis: Optional[int] = -2,
        activation: Optional[AnyStr] = "linear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method = method.lower()
        self.axis = axis
        self.activation = tf.keras.activations.get(activation)

    def get_config(self):
        config = {
            "activation": tf.keras.activations.serialize(self.activation),
            "method": self.method,
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        """
        Apply the layer to the node embeddings in x. These embeddings are either:

          * A list of two tensors of shape (N, M) being the embeddings for each of the nodes in the link,
            where N is the number of links, and M is the node embedding size.
          * A single tensor of shape (..., N, 2, M) where the axis second from last indexes the nodes
            in the link and N is the number of links and M the embedding size.

        """
        # Currently GraphSAGE & HinSage output a list of two tensors being the embeddings
        # for each of the nodes in the link. However, GCN, GAT & other full-batch methods
        # return a tensor of shape (1, N, 2, M).
        # Detect and support both inputs
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("Expecting a list of length 2 for link embedding")
            x0, x1 = x
        elif isinstance(x, tf.Tensor):
            if int(x.shape[self.axis]) != 2:
                raise ValueError(
                    "Expecting a tensor of shape 2 along specified axis for link embedding"
                )
            x0, x1 = tf.unstack(x, axis=self.axis)
        else:
            raise TypeError("Expected a list, tuple, or Tensor as input")

        # Apply different ways to combine the node embeddings to a link embedding.
        if self.method in ["ip", "dot"]:
            out = tf.reduce_sum(x0 * x1, axis=-1, keepdims=True)

        elif self.method == "l1":
            # l1(u,v)_i = |u_i - v_i| - vector of the same size as u,v
            out = tf.abs(x0 - x1)

        elif self.method == "l2":
            # l2(u,v)_i = (u_i - v_i)^2 - vector of the same size as u,v
            out = tf.square(x0 - x1)

        elif self.method in ["mul", "hadamard"]:
            out = tf.multiply(x0, x1)

        elif self.method == "concat":
            out = Concatenate()([x0, x1])

        elif self.method == "avg":
            out = Average()([x0, x1])

        else:
            raise NotImplementedError(
                "{}: the requested method '{}' is not known/not implemented".format(
                    name, edge_embedding_method
                )
            )

        # Apply activation function
        out = self.activation(out)

        return out


def link_inference(
    output_dim: int = 1,
    output_act: AnyStr = "linear",
    edge_embedding_method: AnyStr = "ip",
    clip_limits: Optional[Tuple[float]] = None,
    name: AnyStr = "link_inference",
):
    """
    Defines an edge inference function that takes source, destination node embeddings (node features) as input,
    and returns a numeric vector of output_dim size.

    This function takes as input as either:

     * A list of two tensors of shape (N, M) being the embeddings for each of the nodes in the link,
       where N is the number of links, and M is the node embedding size.
     * A single tensor of shape (..., N, 2, M) where the axis second from last indexes the nodes
       in the link and N is the number of links and M the embedding size.

    Note that the output tensor is flattened before being returned.

    Args:
        output_dim (int): Number of predictor's output units -- desired dimensionality of the output.
        output_act (str), optional: activation function applied to the output, one of "softmax", "sigmoid", etc.,
            or any activation function supported by Keras, see https://keras.io/activations/ for more information.
        edge_embedding_method (str), optional: Name of the method of combining (src,dst) node features or embeddings into edge embeddings.
            One of:
            * 'concat' -- concatenation,
            * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
            * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
            * 'l1' -- L1 operator, :math:`l_1(u,v)_i = |u_i-v_i|`,
            * 'l2' -- L2 operator, :math:`l_2(u,v)_i = (u_i-v_i)^2`,
            * 'avg' -- average, :math:`avg(u,v) = (u+v)/2`.
        clip_limits (Tuple[float]): lower and upper thresholds for LeakyClippedLinear unit on top. If None (not provided),
            the LeakyClippedLinear unit is not applied.
        name (str): optional name of the defined function, used for error logging
            For all methods except 'ip' or 'dot' a dense layer is applied on top of the combined
            edge embedding to transform to a vector of size `output_dim`.

    Returns:
        Function taking edge tensors with src, dst node embeddings (i.e., pairs of (node_src, node_dst) tensors) and
        returning a vector of output_dim length (e.g., edge class probabilities, edge attribute prediction, etc.).
    """

    if edge_embedding_method in ["ip", "dot"] and output_dim != 1:
        warnings.warn(
            "For inner product link method the output_dim will be ignored as it is fixed to be 1.",
            stacklevel=2,
        )
        output_dim = 1

    def edge_function(x):
        le = LinkEmbedding(activation="linear", method=edge_embedding_method)(x)

        # All methods apart from inner product have a dense layer
        # to convert link embedding to the desired output
        if edge_embedding_method in ["ip", "dot"]:
            out = Activation(output_act)(le)
        else:
            out = Dense(output_dim, activation=output_act)(le)

        # Reshape outputs
        out = Reshape((output_dim,))(out)

        if clip_limits:
            out = LeakyClippedLinear(
                low=clip_limits[0], high=clip_limits[1], alpha=0.1
            )(out)
        return out

    print(
        "{}: using '{}' method to combine node embeddings into edge embeddings".format(
            name, edge_embedding_method
        )
    )
    return edge_function


def link_classification(
    output_dim: int = 1,
    output_act: AnyStr = "sigmoid",
    edge_embedding_method: AnyStr = "ip",
):
    """
    Defines a function that predicts a binary or multi-class edge classification output from
    (source, destination) node embeddings (node features).

    This function takes as input as either:

     * A list of two tensors of shape (N, M) being the embeddings for each of the nodes in the link,
       where N is the number of links, and M is the node embedding size.
     * A single tensor of shape (..., N, 2, M) where the axis second from last indexes the nodes
       in the link and N is the number of links and M the embedding size.

    Note that the output tensor is flattened before being returned.

    Args:
        output_dim (int): Number of classifier's output units -- desired dimensionality of the output,
        output_act (str), optional: activation function applied to the output, one of "softmax", "sigmoid", etc.,
            or any activation function supported by Keras, see https://keras.io/activations/ for more information.
        edge_embedding_method (str), optional: Name of the method of combining (src,dst) node features/embeddings into edge embeddings.
            One of:
            * 'concat' -- concatenation,
            * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
            * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
            * 'l1' -- L1 operator, :math:`l_1(u,v)_i = |u_i-v_i|`,
            * 'l2' -- L2 operator, :math:`l_2(u,v)_i = (u_i-v_i)^2`,
            * 'avg' -- average, :math:`avg(u,v) = (u+v)/2`.

    Returns:
        Function taking edge tensors with src, dst node embeddings (i.e., pairs of (node_src, node_dst) tensors) and
        returning logits of output_dim length (e.g., edge class probabilities).
    """

    edge_function = link_inference(
        output_dim=output_dim,
        output_act=output_act,
        edge_embedding_method=edge_embedding_method,
        name="link_classification",
    )

    return edge_function


def link_regression(
    output_dim: int = 1,
    clip_limits: Optional[Tuple[float]] = None,
    edge_embedding_method: AnyStr = "ip",
):
    """
    Defines a function that predicts a numeric edge regression output vector/scalar from
    (source, destination) node embeddings (node features).

    This function takes as input as either:

     * A list of two tensors of shape (N, M) being the embeddings for each of the nodes in the link,
       where N is the number of links, and M is the node embedding size.
     * A single tensor of shape (..., N, 2, M) where the axis second from last indexes the nodes
       in the link and N is the number of links and M the embedding size.

    Note that the output tensor is flattened before being returned.

    Args:
        output_dim (int): Number of classifier's output units -- desired dimensionality of the output,
        clip_limits (tuple): lower and upper thresholds for LeakyClippedLinear unit on top. If None (not provided),
            the LeakyClippedLinear unit is not applied.
        edge_embedding_method (str), optional: Name of the method of combining (src,dst) node features/embeddings into edge embeddings.
            One of:
            * 'concat' -- concatenation,
            * 'ip' or 'dot' -- inner product, :math:`ip(u,v) = sum_{i=1..d}{u_i*v_i}`,
            * 'mul' or 'hadamard' -- element-wise multiplication, :math:`h(u,v)_i = u_i*v_i`,
            * 'l1' -- L1 operator, :math:`l_1(u,v)_i = |u_i-v_i|`,
            * 'l2' -- L2 operator, :math:`l_2(u,v)_i = (u_i-v_i)^2`,
            * 'avg' -- average, :math:`avg(u,v) = (u+v)/2`.

    Returns:
        Function taking edge tensors with src, dst node embeddings (i.e., pairs of (node_src, node_dst) tensors) and
        returning a numeric value (e.g., edge attribute being predicted) constructed according to edge_embedding_method.
    """

    edge_function = link_inference(
        output_dim=output_dim,
        output_act="linear",
        edge_embedding_method=edge_embedding_method,
        clip_limits=clip_limits,
        name="link_regression",
    )

    return edge_function
