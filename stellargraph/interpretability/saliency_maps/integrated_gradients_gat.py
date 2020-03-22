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
from .saliency_gat import GradientSaliencyGAT
import scipy.sparse as sp
from tensorflow.keras import backend as K


class IntegratedGradientsGAT(GradientSaliencyGAT):
    """
    A SaliencyMask class that implements the integrated gradients method.
    """

    def __init__(self, model, generator, node_list):
        self.node_list = list(node_list)
        super().__init__(model, generator)

    def get_integrated_node_masks(
        self,
        node_id,
        class_of_interest,
        X_baseline=None,
        steps=20,
        non_exist_feature=False,
    ):
        """
        This function computes the integrated gradients which measure the importance of each feature to the prediction score of 'class_of_interest'
        for node 'node_id'.

        Args:
        node_id (int): The node ID in the StellarGraph object.
        class_of_interest (int): The  class of interest for which the saliency maps are computed.
        X_baseline: For integrated gradients, X_baseline is the reference X to start with. Generally we should set X_baseline to a all-zero
                                              matrix with the size of the original feature matrix for existing features.
        steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.
        non_exist_feature (bool): Setting it to True allows to compute the importance of features that are 0.
        return (Numpy array): Integrated gradients for the node features.
        """
        node_idx = self.node_list.index(node_id)

        X_val = self.X
        if X_baseline is None:
            if not non_exist_feature:
                X_baseline = np.zeros(X_val.shape)
            else:
                X_baseline = X_val
                X_val = np.ones_like(X_val)
        X_diff = X_val - X_baseline
        total_gradients = np.zeros(X_val.shape)

        for alpha in np.linspace(1.0 / steps, 1, steps):
            X_step = X_baseline + alpha * X_diff
            total_gradients += super().get_node_masks(
                node_idx, class_of_interest, X_val=X_step
            )
        return np.squeeze(total_gradients * X_diff, 0)

    def get_link_importance(
        self, node_id, class_of_interest, steps=20, non_exist_edge=False
    ):
        """
        This function computes the integrated gradients which measure the importance of each edge to the prediction score of 'class_of_interest'
        for node 'node_id'.

        Args:
        node_id (int): The node ID in the StellarGraph object.
        class_of_interest (int): The  class of interest for which the saliency maps are computed.
        steps (int): The number of values we need to interpolate. Generally steps = 20 should give good enough results.\
        non_exist_edge (bool): Setting to True allows the function to get the importance for non-exist edges. This is useful when we want to understand
            adding which edges could change the current predictions. But the results for existing edges are not reliable. Simiarly, setting to False ((A_baseline = all zero matrix))
            could only accurately measure the importance of existing edges.

        return (Numpy array): shape the same with A_val. Integrated gradients for the links.
        """
        node_idx = self.node_list.index(node_id)

        A_val = self.A
        total_gradients = np.zeros(A_val.shape)
        A_diff = (
            A_val
            if not non_exist_edge
            else (np.ones_like(A_val) - np.eye(A_val.shape[0]) - A_val)
        )
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            if self.is_sparse:
                A_val = sp.lil_matrix(A_val)
            tmp = super().get_link_masks(
                alpha, node_idx, class_of_interest, int(non_exist_edge)
            )
            if self.is_sparse:
                tmp = sp.csr_matrix(
                    (tmp, A_val.indices, A_val.indptr), shape=A_val.shape
                ).toarray()
            total_gradients += tmp
        return np.squeeze(np.multiply(total_gradients, A_diff) / steps, 0)

    def get_node_importance(self, node_id, class_of_interest, steps=20):
        """
        The importance of the node is defined as the sum of all the feature importance of the node.

        Args:
            Refer to the parameters in get_integrated_node_masks.

        return (float): Importance score for the node.
        """
        gradients = self.get_integrated_node_masks(
            node_id, class_of_interest, steps=steps
        )
        return np.sum(gradients, axis=1)
