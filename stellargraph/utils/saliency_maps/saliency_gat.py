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


import numpy as np
from tensorflow.keras import backend as K
import scipy.sparse as sp
from stellargraph.mapper import FullBatchSequence
import tensorflow as tf


class GradientSaliencyGAT:
    """
    Class to compute the saliency maps based on the vanilla gradient w.r.t the adjacency and the feature matrix.

    Args:
        model (Keras model object): The differentiable graph model object.
            model.input should contain two tensors:
                - features (Numpy array): The placeholder of the feature matrix.
                - adj (Numpy array): The placeholder of the adjacency matrix.
            model.output (Keras tensor): The tensor of model prediction output.
                This is typically the logit or softmax output.
    """

    def __init__(self, model, generator):
        """
        Args:
            model (Keras model object): The Keras GAT model.
            generator (FullBatchSequence object): The generator from which we extract the feature and adjacency matirx.
        """
        # The placeholders for features and adjacency matrix (model input):
        if not isinstance(generator, FullBatchSequence):
            raise TypeError(
                "The generator supplied has to be an object of FullBatchSequence."
            )
        self.model = model
        # Collect variables for IG
        self.deltas = []
        self.non_exist_edges = []
        for var in model.non_trainable_weights:
            if "ig_delta" in var.name:
                self.deltas.append(var)
            if "ig_non_exist_edge" in var.name:
                self.non_exist_edges.append(var)

        features_t, output_indices_t, adj_t = model.input
        # Placeholder for class prediction (model output):
        output = self.model.output
        self.A = generator.A_dense
        self.X = generator.features
        self.is_sparse = K.is_sparse(adj_t)

    def compute_node_gradients(self, node_mask_tensors):

        for i, x in enumerate(node_mask_tensors):
            if not isinstance(x, tf.Tensor):
                node_mask_tensors[i] = tf.convert_to_tensor(x)

        X_val, out_indices, A_val, _, class_of_interest = node_mask_tensors
        model_input = [X_val, out_indices, A_val]

        with tf.GradientTape() as tape:
            tape.watch(X_val)
            output = self.model(model_input)

            cost_value = K.gather(output[0, 0], class_of_interest)

        node_gradients = tape.gradient(cost_value, X_val)

        return node_gradients

    def compute_link_gradients(self, link_mask_tensors):
        for i, x in enumerate(link_mask_tensors):
            if not isinstance(x, tf.Tensor):
                link_mask_tensors[i] = tf.convert_to_tensor(x)

        X_val, out_indices, A_val, _, class_of_interest = link_mask_tensors
        model_input = [X_val, out_indices, A_val]

        with tf.GradientTape() as tape:
            tape.watch(A_val)
            output = self.model(model_input)
            if self.is_sparse:
                cost_value = (
                    K.gather(K.gather(output, out_indices), class_of_interest),
                )

            else:
                cost_value = K.gather(output[0, 0], class_of_interest)

        link_gradients = tape.gradient(cost_value, A_val)

        return link_gradients

    def set_ig_values(self, delta_value, edge_value):
        """
        Set values of the integrated gradient parameters in all layers of the model.

        Args:
            delta_value: Value of the `delta` parameter
            edge_value: Value of the `non_exist_edges` parameter
        """
        for delta_var in self.deltas:
            K.set_value(delta_var, delta_value)
        for edge_var in self.non_exist_edges:
            K.set_value(edge_var, edge_value)

    def get_node_masks(self, node_id, class_of_interest, X_val=None, A_val=None):
        """
        Args:
            This function computes the saliency maps (gradients) which measure the importance of each feature to the prediction score of 'class_of_interest'
            for node 'node_id'.

            node_id (int): The node ID in the StellarGraph object.
            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            X_val (Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.
            A_val (Numpy array): The adjacency matrix, we do not directly get it from generator to support the integrated gradients computation.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        out_indices = np.array([[node_id]])

        if X_val is None:
            X_val = self.X
        if A_val is None:
            A_val = self.A
        # Execute the function to compute the gradient
        self.set_ig_values(1.0, 0.0)
        if self.is_sparse and not sp.issparse(A_val):
            A_val = sp.lil_matrix(A_val)
        gradients = self.compute_node_gradients(
            [X_val, out_indices, A_val, 0, class_of_interest]
        )
        return gradients[0]

    def get_link_masks(
        self, alpha, node_id, class_of_interest, non_exist_edge, X_val=None, A_val=None
    ):
        """
        This function computes the saliency maps (gradients) which measure the importance of each edge to the prediction score of 'class_of_interest'
        for node 'node_id'.

        Args:
            alpha (float): The path position parameter to support integrated gradient computation.
            node_id (int): The node ID in the StellarGraph object.
            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            non_exist_edge (bool): Setting to True allows the function to get the importance for non-exist edges. This is useful when we want to understand
                adding which edges could change the current predictions. But the results for existing edges are not reliable. Simiarly, setting to False ((A_baseline = all zero matrix))
                could only accurately measure the importance of existing edges.
            X_val (Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.
            A_val (Numpy array): The adjacency matrix, we do not directly get it from generator to support the integrated gradients computation.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        out_indices = np.array([[node_id]])

        if X_val is None:
            X_val = self.X
        if A_val is None:
            A_val = self.A
        # Execute the function to compute the gradient
        self.set_ig_values(alpha, non_exist_edge)
        if self.is_sparse and not sp.issparse(A_val):
            A_val = sp.lil_matrix(A_val)
        gradients = self.compute_link_gradients(
            [X_val, out_indices, A_val, 0, class_of_interest]
        )
        return gradients[0]

    def get_node_importance(self, node_id, class_of_interest, X_val=None, A_val=None):
        """
        For nodes, the saliency mask we get gives us the importance of each features. For visualization purpose, we may
        want to see a summary of the importance for the node. The importance of each node can be defined as the sum of
        all the partial gradients w.r.t its features.

        Args:
            node_id (int): The node ID in the StellarGraph object.
            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            non_exist_edge (bool): Setting to True allows the function to get the importance for non-exist edges. This is useful when we want to understand
                adding which edges could change the current predictions. But the results for existing edges are not reliable. Simiarly, setting to False ((A_baseline = all zero matrix))
                could only accurately measure the importance of existing edges.
            X_val (Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.
            A_val (Numpy array): The adjacency matrix, we do not directly get it from generator to support the integrated gradients computation.        Returns:
        """

        if X_val is None:
            X_val = self.X
        if A_val is None:
            A_val = self.A
        gradients = self.get_node_masks(X_val, A_val, node_id, class_of_interest)[0]
        return np.sum(gradients, axis=1)
