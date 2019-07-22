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


import numpy as np
import keras.backend as K
import scipy.sparse as sp
from stellargraph.mapper.node_mappers import (
    SparseFullBatchNodeSequence,
    FullBatchNodeSequence,
)
from scipy.sparse import csr_matrix


class GradientSaliency(object):
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
        # Set sparse flag from the generator
        self.is_sparse = generator.use_sparse
        self.model = model

        # The placeholders for features and adjacency matrix (model input):
        if self.is_sparse:
            if not isinstance(generator, SparseFullBatchNodeSequence):
                raise TypeError(
                    "The generator supplied has to be an object of SparseFullBatchNodeSequence for sparse adjacency matrix."
                )
            if len(model.input) != 4:
                raise RuntimeError(
                    "Keras model for sparse adjacency is expected to have four inputs"
                )
            self.A = generator.A_values
            self.A_indices = generator.A_indices
            features_t, output_indices_t, adj_indices_t, adj_t = model.input
        else:
            if not isinstance(generator, FullBatchNodeSequence):
                raise TypeError(
                    "The generator supplied has to be an object of FullBatchNodeSequence for dense adjacency matrix."
                )
            if len(model.input) != 3:
                raise RuntimeError(
                    "Keras model for dense adjacency is expected to have three inputs"
                )

            self.A = generator.A_dense
            features_t, output_indices_t, adj_t = model.input

        # Collect variables for IG
        self.deltas = []
        self.non_exist_edges = []
        for var in model.non_trainable_weights:
            if "ig_delta" in var.name:
                self.deltas.append(var)
            if "ig_non_exist_edge" in var.name:
                self.non_exist_edges.append(var)

        # Placeholder for class prediction (model output):
        output = self.model.output
        self.X = generator.features

        # The placeholder for the node index of interest. It is typically the index of the target test node.
        self.node_idx = K.placeholder(shape=(), dtype="int32")

        # The placeholder for the class of interest. One will generally use the winning class.
        self.class_of_interest = K.placeholder(shape=(), dtype="int32")

        # The input tensors for computing the node saliency map
        node_mask_tensors = model.input + [
            K.learning_phase(),  # placeholder for mode (train or test) tense
            self.class_of_interest,
        ]

        # The input tensors for computing the link saliency map
        link_mask_tensors = model.input + [K.learning_phase(), self.class_of_interest]

        # node gradients are the gradients of the output's component corresponding to the
        # class of interest, w.r.t. input features of all nodes in the graph
        self.node_gradients = model.optimizer.get_gradients(
            K.gather(output[0, 0], self.class_of_interest), features_t
        )
        # link gradients are the gradients of the output's component corresponding to the
        # class of interest, w.r.t. all elements of the adjacency matrix
        if self.is_sparse:
            print("adjacency matrix tensor is sparse")
            self.link_gradients = model.optimizer.get_gradients(
                K.gather(output[0, 0], self.class_of_interest), adj_t
            )

        else:
            self.link_gradients = model.optimizer.get_gradients(
                K.gather(output[0, 0], self.class_of_interest), adj_t
            )

        self.compute_link_gradients = K.function(
            inputs=link_mask_tensors, outputs=self.link_gradients
        )
        self.compute_node_gradients = K.function(
            inputs=node_mask_tensors, outputs=self.node_gradients
        )

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

    def get_node_masks(self, node_idx, class_of_interest, X_val=None, A_index=None, A_val=None):
        """
        Args:
            X_val, A_val, node_idx, class_of_interest: The values to feed while computing the gradients.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        if X_val is None:
            X_val = self.X
        if A_index is None and self.is_sparse:
            A_index = self.A_indices
        if A_val is None:
            A_val = self.A
        out_indices = np.array([[node_idx]])
        if self.is_sparse:
            gradients = self.compute_node_gradients(
                [X_val, out_indices, A_index, A_val, 0, class_of_interest]
            )
        # Execute the function to compute the gradient
        else:
            gradients = self.compute_node_gradients(
                [X_val, out_indices, A_val, 0, class_of_interest]
            )
        return gradients[0]

    def get_link_masks(
        self, alpha, node_idx, class_of_interest, non_exist_edge, X_val=None, A_index=None, A_val=None
    ):
        """
        Args:
            X_val, A_val, node_idx, class_of_interest: The values to feed while computing the gradients.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        if X_val is None:
            X_val = self.X
        if A_index is None and self.is_sparse:
            A_index = self.A_indices
        if A_val is None:
            A_val = self.A
        out_indices = np.array([[node_idx]])

        # Execute the function to compute the gradient
        self.set_ig_values(alpha, non_exist_edge)
        if self.is_sparse:
            gradients = self.compute_link_gradients(
                [X_val, out_indices, A_index, A_val, 0, class_of_interest]
            )
        else:
            gradients = self.compute_link_gradients(
                [X_val, out_indices, A_val, 0, class_of_interest]
            )
        if self.is_sparse:
            return csr_matrix((gradients[0][0], (A_index[0, :, 0], A_index[0, :, 1])))
        return np.squeeze(gradients, 0)


    def get_node_importance(
        self, alpha, node_idx, class_of_interest, X_val=None, A_val=None
    ):
        """
        For nodes, the saliency mask we get gives us the importance of each features. For visualization purpose, we may
        want to see a summary of the importance for the node. The importance of each node can be defined as the sum of
        all the partial gradients w.r.t its features.

        Args:
            X_val, A_val, node_idx, class_of_interest: The values to feed while computing the gradients.
        Returns:
            (Numpy array): Each element indicates the importance of a node.
        """
        gradients = self.get_node_masks(
            node_idx, class_of_interest, X_val=None, A_index=None, A_val=None
        )
        return np.sum(gradients, axis=1)
