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
from .misc import deprecated_model_function

from tensorflow.keras.layers import Input, Lambda, Layer, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras import backend as K
import warnings

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


class DeepGraphInfomax:
    """
    A class to wrap stellargraph models for Deep Graph Infomax unsupervised training
    (https://arxiv.org/pdf/1809.10341.pdf).

    Args:
        base_model: the base stellargraph model class
    """

    def __init__(self, base_model):

        if not isinstance(base_model, (GCN, GAT, APPNP, PPNP)):
            raise TypeError(
                f"base_model: expected GCN, GAT, APPNP or PPNP found {type(base_model).__name__}"
            )

        if base_model.multiplicity != 1:
            warnings.warn(
                f"base_model: expected a node model (multiplicity = 1), found a link model (multiplicity = {base_model.multiplicity}). Base model tensors will be constructed as for a node model.",
                stacklevel=2,
            )

        self.base_model = base_model

        self._node_feats = None
        # specific to full batch models
        self._corruptible_inputs_idxs = [0]

        self._discriminator = DGIDiscriminator()

    def in_out_tensors(self):
        """
        A function to create the the keras inputs and outputs for a Deep Graph Infomax model for unsupervised training.

        Note that the tf.nn.sigmoid_cross_entropy_with_logits loss must be used with this model.

        Example::

            dg_infomax = DeepGraphInfoMax(...)
            x_in, x_out = dg_infomax.in_out_tensors()
            model = Model(inputs=x_in, outputs=x_out)
            model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, ...)

        Returns:
            input and output layers for use with a keras model
        """

        x_inp, node_feats = self.base_model.in_out_tensors(multiplicity=1)

        x_corr = [
            Input(batch_shape=x_inp[i].shape) for i in self._corruptible_inputs_idxs
        ]

        # shallow copy normal inputs and replace corruptible inputs with new inputs
        x_in_corr = x_inp.copy()
        for i, x in zip(self._corruptible_inputs_idxs, x_corr):
            x_in_corr[i] = x

        node_feats_corr = self.base_model(x_in_corr)

        summary = tf.keras.activations.sigmoid(GlobalAveragePooling1D()(node_feats))

        scores = self._discriminator([node_feats, summary])
        scores_corrupted = self._discriminator([node_feats_corr, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=2)

        return x_corr + x_inp, x_out

    def embedding_model(self):
        """
        A function to create the the inputs and outputs for an embedding model.

        Returns:
            input and output layers for use with a keras model
        """

        # these tensors should link into the weights that get trained by `build`
        x_emb_in, x_emb_out = self.base_model.in_out_tensors(multiplicity=1)

        squeeze_layer = Lambda(lambda x: K.squeeze(x, axis=0), name="squeeze")
        x_emb_out = squeeze_layer(x_emb_out)

        return x_emb_in, x_emb_out

    build = deprecated_model_function(in_out_tensors, "build")
