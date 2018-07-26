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

"""Link inference functions"""
from typing import AnyStr, Optional, List
from keras.layers import Concatenate, Dense, Lambda, Multiply, Reshape
from keras import backend as K

def link_classifier(
    hidden_src: Optional[List[int]] = None,
    hidden_dst: Optional[List[int]] = None,
    output_dim: int = 1,
    output_act: AnyStr = "sigmoid",
    edge_feature_method: AnyStr = "ip",
):
    """Returns a function that predicts a binary or multi-class edge classification output from
    source, destination node features.

        hidden_src ([list[int]], optional): Hidden sizes for dense layer transforms of source node features.
            If None, no dense transform is applied.
        hidden_dst ([list[int]], optional): Hidden sizes for dense layer transforms of destination node features.
            If None, no dense transform is applied.
        output_dim: (int) Number of classifier's output units (desired dimensionality of the output)
        output_act: (str, optional): output function, one of "softmax", "sigmoid", etc. - this can be user-defined, but must be a Keras function
        edge_feature_method (str, optional): Name of the method of combining (src,dst) node features into edge features.
            One of 'ip' (inner product), 'mul' (element-wise multiplication), and 'concat' (concatenation)

    Returns:
        Function taking edge tensors with src, dst node features (i.e., pairs of (node_src, node_dst) tensors) and
        returning logits of output_dim length.
    """

    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if hidden_src:
            for hid_src in hidden_src:
                x0 = Dense(hid_src, activation="relu")(x0)

        if hidden_dst:
            for hid_dst in hidden_dst:
                x1 = Dense(hid_dst, activation="relu")(x1)

        if edge_feature_method == "ip":
            out = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                [x0, x1]
            )

        elif edge_feature_method == "mul":
            le = Multiply()([x0, x1])
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        elif edge_feature_method == "concat":
            le = Concatenate()([x0, x1])
            out = Dense(output_dim, activation=output_act)(le)
            out = Reshape((output_dim,))(out)

        else:
            raise NotImplementedError(
                "classification_predictor: the requested method '{}' is not known/not implemented".format(
                    edge_feature_method
                )
            )

        return out

    print(
        "Using '{}' method to combine node embeddings into edge embeddings".format(
            edge_feature_method
        )
    )
    return edge_function