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

from . import GCN, GAT, APPNP, PPNP, GraphSAGE, DirectedGraphSAGE

from tensorflow.keras.layers import Input, Lambda, Layer, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras import backend as K
import warnings
import numpy as np

__all__ = ["DeepGraphInfomax", "DGIDiscriminator"]


class DGIDiscriminator(Layer):
    """
    This Layer computes the Discriminator function for Deep Graph Infomax (https://arxiv.org/pdf/1809.10341.pdf).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):

        first_size = input_shapes[0][-1]
        second_size = input_shapes[1][-1]

        self.kernel = self.add_weight(
            shape=(first_size, second_size),
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
            inputs: a list or tuple of tensors with shapes [(batch, N, F1), (batch, F2)] containing the node features and a
                summary feature vector.
        Returns:
            a Tensor with shape (1, N)
        """

        features, summary = inputs

        score = tf.linalg.matvec(features, tf.linalg.matvec(self.kernel, summary),)

        return score


class Readout(Layer):
    """
    This Layer computes the Readout function for Deep Graph Infomax (https://arxiv.org/pdf/1809.10341.pdf).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):

        self.built = True

    def call(self, node_feats):
        """
        Applies the layer to the inputs.

        Args:
            inputs:
        Returns:
            a Tensor with shape (1, N)
        """

        if len(node_feats.shape) == 3:
            summary = tf.reduce_mean(node_feats, axis=1)
        else:
            summary = tf.reduce_mean(node_feats, axis=0)

        summary = tf.math.sigmoid(summary)

        return summary


class DeepGraphInfomax:
    """
    A class to wrap stellargraph models for Deep Graph Infomax unsupervised training
    (https://arxiv.org/pdf/1809.10341.pdf).

    Args:
        base_model: the base stellargraph model class
    """

    def __init__(self, base_model):

        if isinstance(base_model, (GCN, GAT, APPNP, PPNP)):
            self._corruptible_inputs_idxs = [0]
        elif isinstance(base_model, DirectedGraphSAGE):
            self._corruptible_inputs_idxs = np.arange(base_model.max_slots)
        elif isinstance(base_model, GraphSAGE):
            self._corruptible_inputs_idxs = np.arange(base_model.max_hops + 1)
        else:
            raise TypeError(
                f"base_model: expected GCN, GAT, APPNP, PPNP, GraphSAGE,"
                f"or DirectedGraphSAGE, found {type(base_model).__name__}"
            )

        if base_model.multiplicity != 1:
            warnings.warn(
                f"multiplicity: expected the base_model to have a multiplicity of 1, found"
                f" ({self.base_model.multiplicity}). A multiplicity of 1 will be used to construct the base model."
            )

        self.base_model = base_model

        self._node_feats = None
        self._unique_id = f"DEEP_GRAPH_INFOMAX_{id(self)}"

        self._discriminator = DGIDiscriminator()

    def build(self):
        """
        A function to create the the keras inputs and outputs for a Deep Graph Infomax model for unsupervised training.

        Note that the tf.nn.sigmoid_cross_entropy_with_logits loss must be used with this model.

        Example::

            dg_infomax = DeepGraphInfoMax(...)
            x_in, x_out = dg_infomax.build()
            model = Model(inputs=x_in, outputs=x_out)
            model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, ...)

        Returns:
            input and output layers for use with a keras model
        """

        x_inp, node_feats = self.base_model.build(multiplicity=1)

        # identity layer so we can attach a name to the tensor
        node_feats = Lambda(lambda x: x, name=self._unique_id)(node_feats)
        x_corr = [
            Input(batch_shape=x_inp[i].shape) for i in self._corruptible_inputs_idxs
        ]

        # shallow copy normal inputs and replace corruptible inputs with new inputs
        x_in_corr = x_inp.copy()
        for i, x in zip(self._corruptible_inputs_idxs, x_corr):
            x_in_corr[i] = x

        node_feats_corr = self.base_model(x_in_corr)

        summary = Readout()(node_feats)

        scores = self._discriminator([node_feats, summary])
        scores_corrupted = self._discriminator([node_feats_corr, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=1)

        return x_corr + x_inp, x_out

    def embedding_model(self, model):
        """
        A function to create the the inputs and outputs for an embedding model.

        Args:
            model (keras.Model): the base Deep Graph Infomax model with inputs and outputs created from
                DeepGraphInfoMax.build()
        Returns:
            input and output layers for use with a keras model
        """

        try:
            x_emb_out = model.get_layer(self._unique_id).output
        except ValueError:
            raise ValueError(
                f"model: model must be a keras model with inputs and outputs created "
                f"by the build() method of this instance of DeepGraphInfoMax"
            )

        x_emb_in = model.inputs[len(self._corruptible_inputs_idxs) :]

        if len(x_emb_out.shape) == 3:
            squeeze_layer = Lambda(lambda x: K.squeeze(x, axis=0), name="squeeze")
            x_emb_out = squeeze_layer(x_emb_out)

        return x_emb_in, x_emb_out
