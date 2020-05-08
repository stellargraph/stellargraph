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
from .misc import deprecated_model_function
from ..mapper import Node2VecLinkGenerator, Node2VecNodeGenerator


def _require_without_generator(value, name):
    if value is not None:
        return value
    else:
        raise ValueError(
            f"{name}: expected a value for 'node_num' and 'multiplicity' when "
            f"'generator' is not provided, found {name}=None."
        )


class Node2Vec:
    """
    Implementation of the Node2Vec algorithm of A. Grover and J. Leskovec with Keras layers.
    see: https://snap.stanford.edu/node2vec/

    The model minimally requires specification of the embedding size and a generator object.

    Args:
        emb_size (int): The dimension of node embeddings.
        generator (Sequence): A NodeSequence or LinkSequence.
        node_num(int, optional): The number of nodes in the given graph.
        multiplicity (int, optional): The number of nodes to process at a time. This is 1 for a node inference
          and 2 for link inference (currently no others are supported).
    """

    def __init__(self, emb_size, generator=None, node_num=None, multiplicity=None):

        # Get the node_num from the generator if it is given
        self.generator = generator
        if generator is not None:
            self._get_sizes_from_generator(generator)
        else:
            self.input_node_num = _require_without_generator(node_num, "node_num")
            self.multiplicity = _require_without_generator(multiplicity, "multiplicity")

        # Model parameters
        self.emb_size = emb_size

        # Initialise the target embedding layer: input-to-hidden
        target_embedding_initializer = keras.initializers.RandomUniform(
            minval=-1.0, maxval=1.0
        )
        self.target_embedding = Embedding(
            self.input_node_num,
            self.emb_size,
            input_length=1,
            name="target_embedding",
            embeddings_initializer=target_embedding_initializer,
        )

        # Initialise the context embedding layer: hidden-to-output
        context_embedding_initializer = keras.initializers.TruncatedNormal(
            stddev=1.0 / math.sqrt(self.emb_size * 1.0)
        )
        self.context_embedding = Embedding(
            self.input_node_num,
            self.emb_size,
            input_length=1,
            name="context_embedding",
            embeddings_initializer=context_embedding_initializer,
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
            raise ValueError("Node2Vec called on graph with more than one node type.")

    def __call__(self, xin, embedding):
        """
        Construct node representations from node ids through a look-up table.

        Args:
            xin (Keras Tensor): Batch input node ids.
            embedding (str): "target" for target_embedding, "context" for context_embedding

        Returns:
            Output tensor.
        """

        if embedding == "target":
            h_layer = self.target_embedding(xin)
        elif embedding == "context":
            h_layer = self.context_embedding(xin)
        else:
            raise ValueError(
                'wrong embedding argument is supplied: {}, should be "target" or "context"'.format(
                    embedding
                )
            )

        h_layer = Reshape((self.emb_size,))(h_layer)

        return h_layer

    def _node_model(self, embedding="target"):
        """
        Builds a Node2Vec model for node prediction.

        Args:
            embedding (str): "target" for target_embedding, "context" for context_embedding

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

    def _link_model(self):
        """
        Builds a Node2Vec model for link or node pair prediction.

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors for (src, dst) nodes in the node pairs
            and ``x_out`` is a list of output tensors for (src, dst) nodes in the node pairs.

        """
        # Expose input and output sockets of the model, for source node:
        x_inp_src, x_out_src = self._node_model("target")
        x_inp_dst, x_out_dst = self._node_model("context")

        x_inp = [x_inp_src, x_inp_dst]
        x_out = [x_out_src, x_out_dst]
        return x_inp, x_out

    def in_out_tensors(self, multiplicity=None):
        """
        Builds a Node2Vec model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: (x_inp, x_out), where ``x_inp`` contains Keras input tensor(s)
            for the specified Node2Vec model (either node or link/node pair model) and ``x_out`` contains
            model output tensor(s) of shape (batch_size, self.emb_size)

        """
        if multiplicity is None:
            multiplicity = self.multiplicity
        if self.multiplicity == 1:
            return self._node_model()
        elif self.multiplicity == 2:
            return self._link_model()
        else:
            raise ValueError("Currently only multiplicities of 1 and 2 are supported.")

    def default_model(self, flatten_output=True):
        warnings.warn(
            "The .default_model() method is deprecated. Please use .in_out_tensors() method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.build()

    node_model = deprecated_model_function(_node_model, "node_model")
    link_model = deprecated_model_function(_link_model, "link_model")
    build = deprecated_model_function(in_out_tensors, "build")
