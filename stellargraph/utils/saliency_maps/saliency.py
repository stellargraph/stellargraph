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
from tensorflow.keras import backend as K
from scipy.sparse import csr_matrix
from stellargraph.mapper.node_mappers import (
    SparseFullBatchNodeSequence,
    FullBatchNodeSequence,
)


class GradientSaliency:
    """
    Class to compute the saliency maps based on the vanilla gradient w.r.t the adjacency and the feature matrix.

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

        # Extract features from generator
        self.X = generator.features

        # Placeholder for class prediction (model output):
        output = model.output

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
            self.link_gradients = model.optimizer.get_gradients(
                K.gather(output[0, 0], self.class_of_interest), adj_t
            )
            # raise NotImplementedError("Sparse matrix support is not yet implemented")

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

    def get_node_masks(
        self, node_idx, class_of_interest, X_val=None, A_index=None, A_val=None
    ):
        """
        Args:
            node_idx, class_of_interest: The values to feed while computing the gradients.
            X_val, The value of node features, default is obtained from the generator.
            A_val: The values of adjacency matrix while computing the gradients. When the adjacency matrix is sparse, it only contains the non-zero values. The default is obtained from the generator.
            A_index: When the adjacency matrix is sparse, it is the indices of the non-zero values. The default is obtained from the generator.

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
        self, node_idx, class_of_interest, X_val=None, A_index=None, A_val=None
    ):
        """
        Args:
            node_idx, class_of_interest: The values to feed while computing the gradients.
            X_val, The value of node features, default is obtained from the generator.
            A_val: The values of adjacency matrix while computing the gradients. When the adjacency matrix is sparse, it only contains the non-zero values. The default is obtained from the generator.
            A_index: When the adjacency matrix is sparse, it is the indices of the non-zero values. The default is obtained from the generator.
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
        if self.is_sparse:
            # raise NotImplementedError("Sparse matrix support is not yet implemented")
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
        self, node_idx, class_of_interest, X_val=None, A_index=None, A_val=None
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
