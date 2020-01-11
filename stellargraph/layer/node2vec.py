# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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
node2vec

"""
__all__ = ["Node2Vec"]

from tensorflow.keras import Input
from tensorflow.keras.layers import Reshape, Embedding
import math
from tensorflow import keras
import warnings

from ..mapper import Node2VecLinkGenerator, Node2VecNodeGenerator


class Node2Vec:
    """
    Implementation of the node2vec algorithm of A. Grover and J. Leskovec with Keras layers.
    see: https://snap.stanford.edu/node2vec/

    The model minimally requires specification of the embedding size and a generator object.

    Args:
        emb_size (int): The dimension of node embeddings.
        generator (Sequence): A NodeSequence or LinkSequence.
        **kwargs: Parameters node_num(int) and multiplicity(int).
    """

    def __init__(self, emb_size, generator=None, **kwargs):

        # Get the node_num from the generator if it is given
        self.generator = generator
        if generator is not None:
            self._get_sizes_from_generator(generator)
        else:
            self._get_sizes_from_keywords(kwargs)

        # Model parameters
        self.emb_size = emb_size

        # Initialise the embedding layers: input-to-hidden and hidden-to-output
        input_initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
        self.input_embedding = Embedding(
            self.input_node_num,
            self.emb_size,
            input_length=1,
            name="input_embedding",
            embeddings_initializer=input_initializer,
        )

        output_initializer = keras.initializers.TruncatedNormal(
            stddev=1.0 / math.sqrt(self.emb_size * 1.0)
        )
        self.output_embedding = Embedding(
            self.input_node_num,
            self.emb_size,
            input_length=1,
            name="output_embedding",
            embeddings_initializer=output_initializer,
        )

    def _get_sizes_from_generator(self, generator):
        """
        Sets node_num and multiplicity from the generator.
        Args:
             generator: The supplied generator.
        """
        if not isinstance(generator, (Node2VecNodeGenerator, Node2VecLinkGenerator)):
            raise TypeError(
                "Generator should be an instance of Node2VecNodeGenerator or Node2VecLinkGenerator"
            )

        self.multiplicity = generator.multiplicity
        self.input_node_num = generator.graph.number_of_nodes()

        if len(list(generator.graph.node_types)) > 1:
            raise RuntimeError("Node2Vec called on graph with more than one node type.")

    def _get_sizes_from_keywords(self, kwargs):
        """
        Sets node_num and multiplicity from the keywords.
        Args:
             kwargs: The additional keyword arguments.
        """
        try:
            self.input_node_num = kwargs["node_num"]
            self.multiplicity = kwargs["multiplicity"]

        except KeyError:
            raise KeyError(
                "Generator not provided; node_num and multiplicity must be specified."
            )

    def __call__(self, xin, embedding):
        """
        Construct node representations from node ids through a look-up table.

        Args:
            xin (Keras Tensor): Batch input node ids.
            embedding (str): "input" for input_embedding matrix, "output" for output_embedding

        Returns:
            Output tensor.
        """

        if embedding == "input":
            h_layer = self.input_embedding(xin)
        elif embedding == "output":
            h_layer = self.output_embedding(xin)
        else:
            raise ValueError(
                'wrong embedding argument is supplied: {}, should be "input" or "output"'.format(
                    embedding
                )
            )

        h_layer = Reshape((self.emb_size,))(h_layer)
        # K.squeeze(h_layer, axis=0)

        return h_layer

    def node_model(self, embedding="input"):
        """
        Builds a Node2Vec model for node prediction.

        Args:
            embedding (str): "input" for input_embedding, "output" for output_embedding

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a Keras input tensor
            for the Node2Vec model and ``x_out`` is the Keras tensor
            for the Node2Vec model output.

        """
        # Create tensor inputs
        x_inp = Input(shape=(1,))

        # Output from Node2Vec model
        x_out = self(x_inp, embedding)

        return x_inp, x_out

    def link_model(self):
        """
        Builds a Node2Vec model for link or node pair prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) nodes in the node pairs
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs.

        """
        # Expose input and output sockets of the model, for source node:
        x_inp_src, x_out_src = self.node_model("input")
        x_inp_dst, x_out_dst = self.node_model("output")

        x_inp = [x_inp_src, x_inp_dst]
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def build(self):
        """
        Builds a Node2Vec model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` contains Keras input tensor(s)
            for the specified Node2Vec model (either node or link/node pair model) and ``x_out`` contains
            model output tensor(s) of shape (batch_size, self.emb_size)

        """
        if self.multiplicity == 1:
            return self.node_model()
        elif self.multiplicity == 2:
            return self.link_model()
        else:
            raise RuntimeError(
                "Currently only multiplicities of 1 and 2 are supported. Consider using node_model or "
                "link_model method explicitly to build node or link prediction model, respectively."
            )

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method will be deprecated in future versions. "
            "Please use .build() method instead.",
            PendingDeprecationWarning,
        )
        return self.build()
