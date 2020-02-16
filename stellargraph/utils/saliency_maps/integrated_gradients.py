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
The vanilla gradients may not work well for the graph setting. The main reason is that when you compute the vanilla gradients,
you only get the direction of changing at the current state of the graph (i.e., the adjacency matrix and feature matrix). However,
even though the feature values and entries in the adjacency matrix are not continous values, the model (e.g., GCN or GAT) learns
a continous function which may not be linear when a feature or edge value changes discretely. Let's take ReLU(x) as an example, when x
changes from 0 to 1, the output of the function changes from 0 to 1 as well. However, when you compute the gradient of the function
at x = 0, you will get grad(ReLU(x = 0)) = 0 which is obviously not what we want.

Integrated gradients approximates Shapley values by integrating partial gradients w.r.t input features from reference input to the
actual input. Therefore, it could solve the problem we described above and give much better accuracy. It was initially proposed in the paper
"Axiomatic attribution for deep neuron networks" published in ICML'17.
"""

import numpy as np
from tensorflow.keras import backend as K
from scipy.sparse import csr_matrix
import tensorflow as tf
from stellargraph.mapper import SparseFullBatchSequence, FullBatchSequence


class IntegratedGradients:
    """
    A SaliencyMask class that implements the integrated gradients method.
    """

    def __init__(self, model, generator):
        """
        Args:
            model (Keras model object): The differentiable graph model object.
                For a dense model, the model.input should contain two tensors:
                    - features: The placeholder of the feature matrix.
                    - adj: The placeholder of the adjacency matrix.
                For a sparse model, the model.input should contain three tensors:
                    - features: The placeholder of the feature matrix.
                    - adj_index: The placeholder of the adjacency matrix.
                    - adj_values: The placeholder of the adjacency matrix.
                The model.output (Keras tensor) is the tensor of model prediction output.
                    This is typically the logit or softmax output.

        """
        # Set sparse flag from the generator
        self._is_sparse = generator.use_sparse

        if self._is_sparse:
            if not isinstance(generator, SparseFullBatchSequence):
                raise TypeError(
                    "The generator supplied has to be an object of SparseFullBatchSequence for sparse adjacency matrix."
                )
            if len(model.input) != 4:
                raise RuntimeError(
                    "Keras model for sparse adjacency is expected to have four inputs"
                )
            self._adj = generator.A_values
            self._adj_inds = generator.A_indices
        else:
            if not isinstance(generator, FullBatchSequence):
                raise TypeError(
                    "The generator supplied has to be an object of FullBatchSequence for dense adjacency matrix."
                )
            if len(model.input) != 3:
                raise RuntimeError(
                    "Keras model for dense adjacency is expected to have three inputs"
                )

            self._adj = generator.A_dense

        # Extract features from generator
        self._features = generator.features
        self._model = model

    def get_integrated_node_masks(
        self, node_idx, class_of_interest, features_baseline=None, steps=20,
    ):
        """
        Args:
            node_idx: the index of the node to calculate gradients for.
            class_of_interest: the index for the class probability that the gradients will be calculated for.
            features_baseline: For integrated gradients, X_baseline is the reference X to start with. Generally we should set
                X_baseline to a all-zero matrix with the size of the original feature matrix.
            steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        Returns
            (Numpy array): Integrated gradients for the node features.
        """
        if features_baseline is None:
            features_baseline = np.zeros(self._features.shape)

        features_diff = self._features - features_baseline
        total_gradients = np.zeros(self._features.shape)

        for alpha in np.linspace(0, 1, steps):
            features_step = features_baseline + alpha * features_diff

            if self._is_sparse:
                model_input = [
                    features_step,
                    np.array([[node_idx]]),
                    self._adj_inds,
                    self._adj,
                ]
            else:
                model_input = [features_step, np.array([[node_idx]]), self._adj]

            model_input = [tf.convert_to_tensor(x) for x in model_input]
            grads = self._compute_gradients(
                model_input, class_of_interest, wrt=model_input[0]
            )

            total_gradients += grads

        return np.squeeze(total_gradients * features_diff, 0)

    def get_integrated_link_masks(
        self,
        node_idx,
        class_of_interest,
        non_exist_edge=False,
        adj_baseline=None,
        steps=20,
    ):
        """
        Args:
            node_idx: the index of the node to calculate gradients for.
            class_of_interest: the index for the class probability that the gradients will be calculated for.
            non_exist_edge (bool): Setting to True allows the function to get the importance for non-exist edges.
                This is useful when we want to understand adding which edges could change the current predictions.
                But the results for existing edges are not reliable. Simiarly, setting to False ((A_baseline = all zero matrix))
                could only accurately measure the importance of existing edges.
            adj_baseline: For integrated gradients, adj_baseline is the reference adjacency matrix to start with. Generally
                we should set A_baseline to an all-zero matrix or all-one matrix with the size of the original
                A_baseline matrix.
            steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        Returns
            (Numpy array): Integrated gradients for the links.
        """
        if adj_baseline is None:
            if non_exist_edge:
                adj_baseline = np.ones(self._adj.shape)
            else:
                adj_baseline = np.zeros(self._adj.shape)

        adj_diff = self._adj - adj_baseline

        total_gradients = np.zeros_like(self._adj)

        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            adj_step = adj_baseline + alpha * adj_diff

            if self._is_sparse:
                model_input = [
                    self._features,
                    np.array([[node_idx]]),
                    self._adj_inds,
                    adj_step,
                ]
            else:
                model_input = [
                    self._features,
                    np.array([[node_idx]]),
                    adj_step,
                ]

            model_input = [tf.convert_to_tensor(x) for x in model_input]
            grads = self._compute_gradients(
                model_input, class_of_interest, wrt=model_input[-1]
            )

            total_gradients += grads.numpy()

        if self._is_sparse:
            total_gradients = csr_matrix(
                (total_gradients[0], (self._adj_inds[0, :, 0], self._adj_inds[0, :, 1]))
            )
            adj_diff = csr_matrix(
                (adj_diff[0], (self._adj_inds[0, :, 0], self._adj_inds[0, :, 1]))
            )
            total_gradients = total_gradients.multiply(adj_diff) / steps
        else:
            total_gradients = np.squeeze(
                np.multiply(total_gradients, adj_diff) / steps, 0
            )

        return total_gradients

    def get_node_importance(
        self, node_idx, class_of_interest, steps=20,
    ):
        """
        The importance of the node is defined as the sum of all the feature importance of the node.

        Args:
            node_idx: the index of the node to calculate gradients for.
            class_of_interest: the index for the class probability that the gradients will be calculated for.
            steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        return (float): Importance score for the node.
        """

        gradients = self.get_integrated_node_masks(
            node_idx, class_of_interest, steps=steps,
        )

        return np.sum(gradients, axis=-1)

    def _compute_gradients(self, model_input, class_of_interest, wrt):

        class_of_interest = tf.convert_to_tensor(class_of_interest)

        with tf.GradientTape() as tape:

            tape.watch(wrt)

            output = self._model(model_input)

            cost_value = K.gather(output[0, 0], class_of_interest)

        gradients = tape.gradient(cost_value, wrt)

        return gradients
