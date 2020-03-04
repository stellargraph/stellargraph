# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

from . import GCN, GAT, APPNP, PPNP
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
import tensorflow as tf
from tensorflow.keras import backend as K
from ..core.experimental import experimental


__all__ = [
    "fullbatch_infomax_node_model",
    "fullbatch_infomax_embedding_model",
]

_NODE_FEATS = "GCN_INFO_MAX_NODE_FEATURES"


class Discriminator(Layer):
    """
    This Layer computes the Discriminator function for Deep Graph Infomax (https://arxiv.org/pdf/1809.10341.pdf).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):

        self.kernel = self.add_weight(
            shape=(input_shapes[0][1], input_shapes[0][1]),
            initializer="glorot_uniform",
            name="kernel",
            regularizer=None,
            constraint=None,
        )
        self.built = True

    def call(self, inputs):
        """
        Applies the layer to the inputs.

        Args:
            inputs: a list of tensors with shapes [(N, F), (F,)] containing the node features and summary feature
                vector.
        """
        features, summary = inputs

        score = tf.linalg.matvec(tf.linalg.matmul(features, self.kernel), summary)

        return score


@experimental(reason="lack of unit tests", issues=[1003])
def fullbatch_infomax_node_model(base_model):
    """
    A function to create the the inputs and outputs for a Deep Graph Infomax model for unsupervised training.

    Args:
        base_model: the base stellargraph model class

    Returns:
        input and output layers for use with a keras model
    """

    if not isinstance(base_model, (GCN, GAT, APPNP, PPNP)):
        raise TypeError(
            f"base_model: expected GCN, GAT, APPNP or PPNP found {type(base_model).__name__}"
        )

    # Inputs for features
    x_t = Input(batch_shape=(1, base_model.n_nodes, base_model.n_features))
    # Inputs for shuffled features
    x_corr = Input(batch_shape=(1, base_model.n_nodes, base_model.n_features))

    out_indices_t = Input(batch_shape=(1, None), dtype="int32")

    # Create inputs for sparse or dense matrices
    if base_model.use_sparse:
        # Placeholders for the sparse adjacency matrix
        A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
        A_values_t = Input(batch_shape=(1, None))
        A_placeholders = [A_indices_t, A_values_t]

    else:
        # Placeholders for the dense adjacency matrix
        A_m = Input(batch_shape=(1, base_model.n_nodes, base_model.n_nodes))
        A_placeholders = [A_m]

    x_inp = [x_corr, x_t, out_indices_t] + A_placeholders

    node_feats = base_model([x_t, out_indices_t] + A_placeholders)
    node_feats = Lambda(lambda x: K.squeeze(x, axis=0), name=_NODE_FEATS,)(node_feats)

    node_feats_corrupted = base_model([x_corr, out_indices_t] + A_placeholders)
    node_feats_corrupted = Lambda(lambda x: K.squeeze(x, axis=0))(node_feats_corrupted)

    summary = Lambda(lambda x: tf.math.sigmoid(tf.math.reduce_mean(x, axis=0)))(
        node_feats
    )

    discriminator = Discriminator()
    scores = discriminator([node_feats, summary])
    scores_corrupted = discriminator([node_feats_corrupted, summary])

    x_out = tf.stack([scores, scores_corrupted], axis=1)

    x_out = K.expand_dims(x_out, axis=0)
    return x_inp, x_out


@experimental(reason="lack of unit tests", issues=[1003])
def fullbatch_infomax_embedding_model(info_max_model):
    """
    A function to create the the inputs and outputs for an embedding model.

    Args:
        info_max_model (keras.Model): the base Deep Graph Infomax model
    Returns:
        input and output layers for use with a keras model
    """
    x_emb_in = info_max_model.inputs[1:]
    x_emb_out = info_max_model.get_layer(_NODE_FEATS).output

    return x_emb_in, x_emb_out
