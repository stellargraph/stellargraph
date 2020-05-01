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
from .misc import deprecated_model_function
from ..mapper import CorruptedGenerator

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
            inputs: a list or tuple of tensors with shapes ``[(1, N, F), (1, F)]`` for full batch methods and shapes
                ``[(B, F), (F,)]`` for sampled node methods, containing the node features and a summary feature vector.
                Where ``N`` is the number of nodes in the graph, ``F`` is the feature dimension, and ``B`` is the batch size.
        Returns:
            a Tensor with shape ``(1, N)`` for full batch methods and shape ``(B,)`` for sampled node methods.
        """

        features, summary = inputs

        score = tf.linalg.matvec(features, tf.linalg.matvec(self.kernel, summary),)

        return score


class DGIReadout(Layer):
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
            node_feats: a tensor containing the batch node features from the base model. This has shape `(1, N, F)`
                for full batch methods and shape `(B, F)` for sampled node methods. Where `N` is the number of nodes
                in the graph, `F` is the feature dimension, and `B` is the batch size.

        Returns:
            a Tensor with shape `(1, F)` for full batch methods and shape `(F,)` for sampled node methods.
        """

        summary = tf.reduce_mean(node_feats, axis=-2)
        summary = tf.math.sigmoid(summary)

        return summary


class DeepGraphInfomax:
    """
    A class to wrap stellargraph models for Deep Graph Infomax unsupervised training
    (https://arxiv.org/pdf/1809.10341.pdf).

    Args:
        base_model: the base stellargraph model class
    """

    def __init__(self, base_model, corrupted_generator=None):

        if corrupted_generator is None:
            warnings.warn(
                "The 'corrupted_generator' parameter should be set to an instance of `CorruptedGenerator`, because the support for specific algorithms is being replaced by a more general form",
                DeprecationWarning,
                stacklevel=2,
            )
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
        elif not isinstance(corrupted_generator, CorruptedGenerator):
            raise TypeError(
                f"corrupted_generator: expected a CorruptedGenerator, found {type(corrupted_generator).__name__}"
            )
        else:
            self._corruptible_inputs_idxs = [
                idx
                for group in corrupted_generator.corrupt_index_groups
                for idx in group
            ]

        self.base_model = base_model

        self._node_feats = None
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

        x_inp, node_feats = self.base_model.in_out_tensors()

        x_corr = [
            Input(batch_shape=x_inp[i].shape) for i in self._corruptible_inputs_idxs
        ]

        # shallow copy normal inputs and replace corruptible inputs with new inputs
        x_in_corr = x_inp.copy()
        for i, x in zip(self._corruptible_inputs_idxs, x_corr):
            x_in_corr[i] = x

        node_feats_corr = self.base_model(x_in_corr)

        summary = DGIReadout()(node_feats)

        scores = self._discriminator([node_feats, summary])
        scores_corrupted = self._discriminator([node_feats_corr, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=-1)

        return x_corr + x_inp, x_out

    def embedding_model(self):
        """
        Deprecated: use ``base_model.in_out_tensors`` instead. Deep Graph Infomax just trains the base model,
        and the model behaves as usual after training.
        """

        warnings.warn(
            f"The 'embedding_model' method is deprecated, use 'base_model.in_out_tensors' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # these tensors should link into the weights that get trained by `build`
        x_emb_in, x_emb_out = self.base_model.in_out_tensors()

        # squeeze out batch dim of full batch models
        if len(x_emb_out.shape) == 3:
            squeeze_layer = Lambda(lambda x: K.squeeze(x, axis=0), name="squeeze")
            x_emb_out = squeeze_layer(x_emb_out)

        return x_emb_in, x_emb_out

    build = deprecated_model_function(in_out_tensors, "build")
