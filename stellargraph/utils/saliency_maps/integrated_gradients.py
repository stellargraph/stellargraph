# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
from .saliency import GradientSaliency
import scipy.sparse as sp


class IntegratedGradients(GradientSaliency):
    """
    A SaliencyMask class that implements the integrated gradients method.
    """

    def __init__(self, model):
        super().__init__(model)

    def get_integrated_node_masks(
        self, X_val, A_val, node_idx, class_of_interest, X_baseline=None, steps=20
    ):
        """
        Args:
        X_val, A_val, node_idx, class_of_interest: The feature matrix, adjacency matrix, target node index and class of interest
                                                   which are used to compute the vanilla gradients.
        X_baseline: For integrated gradients, X_baseline is the reference X to start with. Generally we should set X_baseline to a all-zero
                                              matrix with the size of the original feature matrix.
        steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.

        return (Numpy array): Integrated gradients for the node features.
        """
        if X_baseline is None:
            X_baseline = np.zeros(X_val.shape)
        X_diff = X_val - X_baseline

        total_gradients = np.zeros(X_val.shape)
        for alpha in np.linspace(0, 1, steps):
            X_step = X_baseline + alpha * X_diff
            total_gradients += self.get_node_masks(
                X_step, A_val, node_idx, class_of_interest
            )
        return np.squeeze(total_gradients * X_diff, 0)

    def get_integrated_link_masks(
        self,
        X_val,
        A_val,
        node_idx,
        class_of_interest,
        A_baseline=None,
        steps=20,
        non_exist_edge=False,
    ):
        """
        Args:
        X_val, A_val, node_idx, class_of_interest: The feature matrix, adjacency matrix, target node index and class of interest
                                                   which are used to compute the vanilla gradients.
        X_baseline: For integrated gradients, X_baseline is the reference X to start with. Generally we should set X_baseline to a all-zero
                                              matrix with the size of the original adjacency matrix.
        steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.\

        non_exist_edge (bool): Setting to True allows the function to get the importance for non-exist edges. This is useful when we want to understand
            adding which edges could change the current predictions. But the results for existing edges are not reliable. Simiarly, setting to False ((A_baseline = all zero matrix))
            could only accurately measure the importance of existing edges.

        return (Numpy array): shape the same with A_val. Integrated gradients for the links.
        """
        if A_baseline is None:
            if non_exist_edge:
                A_baseline = A_val
                A_val = np.ones(A_baseline.shape)
            else:
                A_baseline = np.zeros(A_val.shape)
        A_diff = A_val - A_baseline

        total_gradients = np.zeros(A_val.shape)
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            A_step = A_baseline + alpha * A_diff
            if self.is_sparse:
                A_step = sp.lil_matrix(A_step)
                A_csr = sp.csr_matrix(A_step)
            tmp = self.get_link_masks(X_val, A_step, node_idx, class_of_interest)
            if self.is_sparse:
                tmp = sp.csr_matrix(
                    (tmp, A_csr.indices, A_csr.indptr), shape=A_csr.shape
                ).toarray()
            total_gradients += tmp

        return np.squeeze(np.multiply(total_gradients, A_diff) / steps, 0)

    def get_node_importance(self, X_val, A_val, node_idx, class_of_interest, steps=20):
        """
        The importance of the node is defined as the sum of all the feature importance of the node.

        Args:
            Refer to the parameters in get_integrated_node_masks.

        return (float): Importance score for the node.
        """
        gradients = self.get_integrated_node_masks(
            X_val, A_val, node_idx, class_of_interest, steps=steps
        )

        return np.sum(gradients, axis=-1)
