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
        self.is_sparse = generator.use_sparse

        if self.is_sparse:
            if not isinstance(generator, SparseFullBatchSequence):
                raise TypeError(
                    "The generator supplied has to be an object of SparseFullBatchSequence for sparse adjacency matrix."
                )
            if len(model.input) != 4:
                raise RuntimeError(
                    "Keras model for sparse adjacency is expected to have four inputs"
                )
            self.A = generator.A_values
            self.A_indices = generator.A_indices
        else:
            if not isinstance(generator, FullBatchSequence):
                raise TypeError(
                    "The generator supplied has to be an object of FullBatchSequence for dense adjacency matrix."
                )
            if len(model.input) != 3:
                raise RuntimeError(
                    "Keras model for dense adjacency is expected to have three inputs"
                )

            self.A = generator.A_dense

        # Extract features from generator
        self.X = generator.features
        self.model = model

    def get_integrated_node_masks(
        self,
        node_idx,
        class_of_interest,
        X_baseline=None,
        steps=20,
    ):
        """
        Args:
            node_idx: the index of the node to calculate gradients for.
            class_of_interest: the index for the class probability that the gradients will be calculated for.
            X_baseline: For integrated gradients, X_baseline is the reference X to start with. Generally we should set
                X_baseline to a all-zero matrix with the size of the original feature matrix.
            steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        Returns
            (Numpy array): Integrated gradients for the node features.
        """
        if X_baseline is None:
            X_baseline = np.zeros(self.X.shape)
        X_diff = self.X - X_baseline

        total_gradients = np.zeros(self.X.shape)
        for alpha in np.linspace(0, 1, steps):
            X_step = X_baseline + alpha * X_diff
            if self.is_sparse:
                grads = self._compute_gradients(
                    [X_step, np.array([[node_idx]]), self.A_indices, self.A, class_of_interest],
                    variable='nodes',
                )
            else:
                grads = self._compute_gradients(
                    [X_step, np.array([[node_idx]]), self.A, class_of_interest],
                    variable='nodes',
                )

            total_gradients += grads

        return np.squeeze(total_gradients * X_diff, 0)

    def get_integrated_link_masks(
        self,
        node_idx,
        class_of_interest,
        non_exist_edge=False,
        A_baseline=None,
        steps=20,
    ):
        """
        Args:
            node_idx: the index of the node to calculate gradients for.
            class_of_interest: the index for the class probability that the gradients will be calculated for.
            A_baseline: For integrated gradients, A_baseline is the reference adjacency matrix to start with. Generally
                we should set A_baseline to an all-zero matrix or all-one matrix with the size of the original
                A_baseline matrix.
            steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        return (Numpy array): shape the same with A_val. Integrated gradients for the links.
        """
        if A_baseline is None:
            if non_exist_edge:
                A_baseline = np.ones(self.A.shape)
            else:
                A_baseline = np.zeros(self.A.shape)

        A_diff = self.A - A_baseline

        total_gradients = np.zeros_like(self.A)

        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            A_step = A_baseline + alpha * A_diff

            # TODO: what is 0?
            if self.is_sparse:
                grads = self._compute_gradients(
                    [self.X, np.array([[node_idx]]), self.A_indices, A_step, 0, class_of_interest],
                    variable="links"
                )
            else:
                grads = self._compute_gradients(
                    [self.X, np.array([[node_idx]]), A_step, 0, class_of_interest],
                    variable="links"
                )

            total_gradients += grads.numpy()

        if self.is_sparse:
            total_gradients = csr_matrix(
                (total_gradients[0], (self.A_indices[0, :, 0], self.A_indices[0, :, 1]))
            )
            A_diff = csr_matrix((A_diff[0], (self.A_indices[0, :, 0], self.A_indices[0, :, 1])))
            total_gradients = total_gradients.multiply(A_diff) / steps
        else:
            total_gradients = np.squeeze(
                np.multiply(total_gradients, A_diff) / steps, 0
            )

        return total_gradients

    def get_node_importance(
        self,
        node_idx,
        class_of_interest,
        steps=20,
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
            node_idx,
            class_of_interest,
            steps=steps,
        )

        return np.sum(gradients, axis=-1)

    def _compute_gradients(self, mask_tensors, variable):

        for i, x in enumerate(mask_tensors):
            if not isinstance(x, tf.Tensor):
                mask_tensors[i] = tf.convert_to_tensor(x)

        if self.is_sparse:
            (
                features_t,
                output_indices_t,
                adj_indices_t,
                adj_t,
                class_of_interest,
            ) = mask_tensors
            model_input = [features_t, output_indices_t, adj_indices_t, adj_t]

        else:
            (
                features_t,
                output_indices_t,
                adj_t,
                _,
                class_of_interest,
            ) = mask_tensors
            model_input = [features_t, output_indices_t, adj_t]

        with tf.GradientTape() as tape:
            if variable == 'nodes':
                tape.watch(features_t)
            elif variable == 'links':
                tape.watch(adj_t)

            output = self.model(model_input)

            cost_value = K.gather(output[0, 0], class_of_interest)

        if variable == 'nodes':
            gradients = tape.gradient(cost_value, features_t)
        elif variable == 'links':
            gradients = tape.gradient(cost_value, adj_t)

        return gradients