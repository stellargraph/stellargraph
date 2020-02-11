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


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape, Embedding

class ComplExScore(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        s_re, s_im, r_re, r_im, o_re, o_im = inputs

        score = tf.reduce_sum(r_re * s_re * o_re + r_re * s_im * o_im + r_im * s_re * o_im - r_im * s_im * o_re, axis=2)

        return score

class ComplEx:
    def __init__(self, num_nodes, num_edge_types, k, embedding_initializer=None, embedding_regularizer=None):
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.k = k
        self.embedding_initializer = initializers.get(embedding_initializer)
        self.embedding_regularizer = regularizers.get(embedding_regularizer)

    def _embed(self, count, name):
        return Embedding(count, self.k, name=name, embeddings_initializer=self.embedding_initializer, embeddings_regularizer=self.embedding_regularizer)

    def __call__(self, x):
        s_iloc, r_iloc, o_iloc = x

        node_embeddings_real = self._embed(self.num_nodes, "NODE_REAL")
        node_embeddings_imag = self._embed(self.num_nodes, "NODE_IMAG")
        edge_type_embeddings_real = self._embed(self.num_nodes, "EDGE_TYPE_REAL")
        edge_type_embeddings_imag = self._embed(self.num_nodes, "EDGE_TYPE_IMAG")

        s_re = node_embeddings_real(s_iloc)
        s_im = node_embeddings_imag(s_iloc)

        r_re = edge_type_embeddings_real(r_iloc)
        r_im = edge_type_embeddings_imag(r_iloc)

        o_re = node_embeddings_real(o_iloc)
        o_im = node_embeddings_imag(o_iloc)

        scoring = ComplExScore()

        return scoring([s_re, s_im, r_re, r_im, o_re, o_im])

    def build(self):
        # FIXME
        s_iloc = Input(batch_shape=(1, None))
        r_iloc = Input(batch_shape=(1, None))
        o_iloc = Input(batch_shape=(1, None))

        x_inp = [s_iloc, r_iloc, o_iloc]
        x_out = self(x_inp)

        return x_inp, x_out
