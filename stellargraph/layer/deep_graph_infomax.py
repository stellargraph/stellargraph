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
from ..core.experimental import experimental

from tensorflow.keras.layers import Input, Dense, Lambda, Layer
import tensorflow as tf
from tensorflow.linalg import matmul
from tensorflow.math import reduce_mean, sigmoid
from tensorflow.keras import backend as K

__all__ = ["InfoMax"]

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

        score = matmul(features, matmul(self.kernel, summary))

        return score


@experimental(reason="lack of unit tests", issues=[1003])
class InfoMax:
    """
    A class to wrap stellargraph models for Deep Graph Infomax unsupervised training
    (https://arxiv.org/pdf/1809.10341.pdf).

    Args:
        base_model: the base stellargraph model class
    """

    _NODE_FEATS = "INFO_MAX_NODE_FEATURES"

    def __init__(self, base_model):

        if not isinstance(base_model, (GCN, GAT, APPNP, PPNP)):
            raise TypeError(
                f"base_model: expected GCN, GAT, APPNP or PPNP found {type(base_model).__name__}"
            )

        self.base_model = base_model

        # specific to full batch models
        self.corruptible_inputs_idxs = [0]

    def unsupervised_node_model(self):
        """
        A function to create the the inputs and outputs for a Deep Graph Infomax model for unsupervised training.

        Returns:
            input and output layers for use with a keras model
        """
        x_inp, node_feats = self.base_model.build(multiplicity=1)
        x_corr = [
            Input(batch_shape=x_inp[i].shape) for i in self.corruptible_inputs_idxs
        ]

        # shallow copy normal inputs and replace corruptible inputs with new inputs
        x_in_corr = [x for x in x_inp]
        for i, x in zip(self.corruptible_inputs_idxs, x_corr):
            x_in_corr[i] = x

        node_feats_corr = self.base_model(x_in_corr)

        # squeezing is specific to full batch models
        node_feats = Lambda(lambda x: K.squeeze(x, axis=0), name=self._NODE_FEATS)(
            node_feats
        )
        node_feats_corrupted = Lambda(lambda x: K.squeeze(x, axis=0))(node_feats_corr)

        summary = Lambda(lambda x: sigmoid(reduce_mean(x, axis=0)))(
            node_feats
        )

        discriminator = Discriminator()
        scores = discriminator([node_feats, summary])
        scores_corrupted = discriminator([node_feats_corrupted, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=1)

        x_out = K.expand_dims(x_out, axis=0)
        return x_corr + x_inp, x_out

    def embedding_model(self, model):
        """
        A function to create the the inputs and outputs for an embedding model.

        Args:
            model (keras.Model): the base Deep Graph Infomax model
        Returns:
            input and output layers for use with a keras model
        """
        x_emb_in = model.inputs[len(self.corruptible_inputs_idxs) :]
        x_emb_out = model.get_layer(self._NODE_FEATS).output

        return x_emb_in, x_emb_out
