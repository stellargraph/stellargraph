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
import scipy.sparse as sp
from stellargraph.mapper.sequences import NodeSequence


class GradientSaliencyGSAGE(object):
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
            generator (FullBatchNodeSequence object): The generator from which we extract the feature and adjacency matirx.
        """
        # The placeholders for features and adjacency matrix (model input):
        if not isinstance(generator, NodeSequence):
            raise TypeError(
                "The generator supplied has to be an object of NodeSequence."
            )
        self.model = model
        self.target_gen = generator
        self.num_samples = generator.num_samples
        self.batch_gen()

        #Collect gradient for ig
        self.link_weight = []
        for var in model.non_trainable_weights:
            if "ig_link_weight" in var.name:
                self.link_weight.append(var)

        features_t = self.model.input
        # Placeholder for class prediction (model output):
        output = self.model.output

        # The placeholder for the node index of interest. It is typically the index of the target test node.
        self.node_id = K.placeholder(shape=(), dtype="int32")

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
            K.gather(output[0], self.class_of_interest), features_t
        )

        # link gradients are the gradients of the output's component corresponding to the
        # class of interest, w.r.t. all elements of the adjacency matrix

        self.link_gradients = model.optimizer.get_gradients(
            K.gather(output[0], self.class_of_interest), self.link_weight
        )

        self.compute_link_gradients = K.function(
            inputs=link_mask_tensors, outputs=self.link_gradients
        )

        self.compute_node_gradients = K.function(
            inputs=node_mask_tensors, outputs=self.node_gradients
        )

    def set_ig_values(self, delta_value):
        """
        Set values of the integrated gradient parameters in all layers of the model.

        Args:
            delta_value: Value of the `delta` parameter
        """
        for delta_var in self.link_weight:
            K.set_value(delta_var, np.ones(delta_var.shape) * delta_value)

    def get_node_masks(self, node_id, class_of_interest, X_val=None, A_val=None):
        """
        Args:
            This function computes the saliency maps (gradients) which measure the importance of each feature to the prediction score of 'class_of_interest'
            for node 'node_id'.

            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            X_val (Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.
            A_val (Numpy array): The adjacency matrix, we do not directly get it from generator to support the integrated gradients computation.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        if X_val is None:
            X_val = self.X
        if A_val is None:
            A_val = self.A

        # Execute the function to compute the gradient [[head],[1-hop],[2-hops]]
        batch_gradients = self.compute_node_gradients(
            [X_val, 0, class_of_interest]
        )

        return self.inplist2arr(batch_gradients)

    def get_link_masks(
        self, alpha, class_of_interest, non_exist_edge, X_val=None, A_val=None
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
        if X_val is None:
            X_val = self.X
        if A_val is None:
            A_val = self.A
        # Execute the function to compute the gradient
        self.set_ig_values(alpha)
        gradients = self.compute_link_gradients(
            [X_val, 0, class_of_interest]
        )
        ex_gradients = []
        for i in gradients:
            tmp = np.sum(i,axis=1)
            ex_gradients.append(np.expand_dims(tmp,axis=0))
        return self.inplist2arr(ex_gradients)

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
        gradients = self.get_node_masks(0, class_of_interest)
        gradlist = self.arr2inplist(gradients)
        gradients_dict = self.node_gradlist2dict(gradlist, self.A)

        return gradients_dict

    ### batch generation
    def batch_gen(self):
        assert self.target_gen.data_size == 1, "data_size for the target generator must be 1"
        self.target_gen.__getitem__(0)
        self.X = self.target_gen.batch_features
        self.A = self.target_gen.batch_nodes_indices
        self.L = self.link_gen()

    def link_gen(self):
        link_list = []
        node_list = self.A
        num_samples = self.num_samples

        link_list.append([(int(node_list[0][0,0]),int(node_list[0][0,0]))])
        for i in range(len(node_list) - 1):
            tmp = []
            for j in range(node_list[i].shape[1]):
                for k in range(num_samples[i]):
                    tmp_tuple = (int(node_list[i][0,j]), int(node_list[i+1][0,j*num_samples[i] + k]))
                    tmp.append(tmp_tuple)
            link_list.append(tmp)
        return link_list


    ### formats for Gsage
    def node_gradlist2dict(self, batch_gradients, batch_index):
        from collections import Counter
        gradients_counter = Counter()
        for i in range(len(batch_index)):
            index = batch_index[i].squeeze(0)
            gradients = batch_gradients[i].squeeze(0)
            for j in range(index.shape[0]):
                gradients_counter.update(Counter(dict(zip([index[j]], [gradients[j]]))))
        gradients_dict = dict(gradients_counter)
        return gradients_dict

    def link_gradlist2dict(self, link_gradients, link_index):
        from collections import Counter
        gradients_counter = Counter()
        for i in range(len(link_index)):
            for j in range(len(link_index[i])):
                gradients_counter.update(Counter(dict(zip([link_index[i][j]], [link_gradients[i][0][j]]))))
        gradients_dict = dict(gradients_counter)
        return gradients_dict


    def inplist2arr(self, inputlist):
        for i in range(len(inputlist)):
            if i == 0:
                arr = inputlist[i]
            else:
                arr = np.concatenate((arr,inputlist[i]), axis=1)
        return arr

    def arr2inplist(self, arr):
        inplist = [arr[:,0:1]]
        start_idx = 1
        layer_num = 1
        for i in self.num_samples:
            layer_num *= i
            end_idx = start_idx + layer_num
            inplist.append(arr[:,start_idx:end_idx])
            start_idx = end_idx
        return inplist
