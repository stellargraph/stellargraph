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
    Class to compute the saliency maps based on the vanilla gradient w.r.t the feature list.
    GraphSAGE takes a list of features as input, whose length depends on the depth of its sampled subgraph.
    We add a link_weight layer before the input was sent to the core of GraphSAGE to inject link importance.

    Args:
        model (Keras model object): The differentiable graph model object.
            model.input should contain one tensor:
                - features (List of Numpy array): The placeholder of the feature list.
            model.output (Keras tensor): The tensor of model prediction output.
                This is typically the logit or softmax output.
    """

    def __init__(self, model, generator):
        """
        Args:
            model (Keras model object): The Keras GSAGE model.
            generator (NodeSequence object): The generator from which we generate a sampled subgraph and extract the feature list.
                                            Notice that target_nid depends on the input generator.
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

        # Collect variable for setting link weight when compute ig_link_importance:
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
        Set link weight to be delta_value of the model.

        Args:
            delta_value: Value of the `delta` parameter
        """
        for delta_var in self.link_weight:
            K.set_value(delta_var, np.ones(delta_var.shape) * delta_value)

    def get_node_masks(self, class_of_interest, X_val=None, A_val=None):
        """
        Args:
            This function computes the saliency maps (gradients) which measure the importance of each feature to the prediction score of 'class_of_interest'
            for target node.

            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            X_val (List of Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.

        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the nodes.
        """
        if X_val is None:
            X_val = self.X

        # Execute the function to compute the gradients
        # Output gradients have same size as input features: [head_array,1-hop_array,2-hops_array ...]
        batch_gradients = self.compute_node_gradients(
            [X_val, 0, class_of_interest]
        )

        # Format the output gradients to a single array by concatenating them.
        return self.inplist2arr(batch_gradients)

    def get_link_masks(
        self, alpha, class_of_interest, non_exist_edge=False, X_val=None, A_val=None
    ):
        """
        This function computes the saliency maps (gradients) which measure the importance of each edge to the prediction score of 'class_of_interest'
        for the head_node.

        Args:
            alpha (float): The path position parameter to support integrated gradient computation.
            class_of_interest (int): The  class of interest for which the saliency maps are computed.
            X_val (Numpy array): The feature matrix, we do not directly get it from generator to support the integrated gradients computation.
        Returns:
            gradients (Numpy array): Returns a vanilla gradient mask for the links.
        """
        if X_val is None:
            X_val = self.X

        # Execute the function to compute the gradient
        self.set_ig_values(alpha)
        gradients = self.compute_link_gradients(
            [X_val, 0, class_of_interest]
        )

        # Expand a batch_size dim for link gradients to make it consistent with node gradients
        expanded_gradients = []
        for link_gradient in gradients:
            tmp = np.sum(link_gradient,axis=1)
            expanded_gradients.append(np.expand_dims(tmp,axis=0))
        return self.inplist2arr(expanded_gradients)



    def batch_gen(self):
        """
            This function is used to generate a subgraph generated by GraphsageGenerator.
            It samples by calling target_gen.__getitem__(0) and records the corresponding features, node_ids and link_ids info.
            features will be the input for gradients computing.
            node_ids will be the keys of the node_importance Counter/dict.
            link_ids will be the keys of the link_importance Counter/dict.
        """
        if self.target_gen.data_size != 1:
            raise ValueError(
                "data_size for the target generator must be 1"
            )

        self.target_gen.__getitem__(0)
        self.X = self.target_gen.batch_features
        self.A = self.target_gen.batch_nodes_indices
        self.L = self.link_gen()

    def link_gen(self):
        """
            This function generates link_ids based on the node_ids.
        """
        link_list = []
        node_list = self.A
        num_samples = self.num_samples
        head_node_id= int(node_list[0][0,0])
        link_list.append([(head_node_id,head_node_id)])
        for i in range(len(node_list) - 1):
            tmp = []
            node_list_cur_layer = node_list[i]
            node_list_next_layer = node_list[i+1]
            for j in range(node_list_cur_layer.shape[1]):
                cur_node_id = int(node_list_cur_layer[0,j])
                for k in range(num_samples[i]):
                    next_node_id = int(node_list_next_layer[0,j*num_samples[i] + k])
                    tmp_tuple = (cur_node_id, next_node_id)
                    tmp.append(tmp_tuple)
            link_list.append(tmp)
        return link_list


    def node_gradlist2dict(self, node_gradients, node_index):
        """
            This function covert the node gradients list to a dict {node_id: gradient}.
        """
        from collections import Counter
        gradients_counter = Counter()
        for i in range(len(node_index)):
            index = node_index[i].squeeze(0)
            gradients = node_gradients[i].squeeze(0)
            for j in range(index.shape[0]):
                gradients_counter.update(Counter(dict(zip([index[j]], [gradients[j]]))))
        gradients_dict = dict(gradients_counter)
        return gradients_dict

    def link_gradlist2dict(self, link_gradients, link_index):
        """
            This function covert the node gradients list to a dict {link_id: gradient}.
        """
        from collections import Counter
        gradients_counter = Counter()
        for i in range(len(link_index)):
            for j in range(len(link_index[i])):
                gradients_counter.update(Counter(dict(zip([link_index[i][j]], [link_gradients[i][0][j]]))))
        gradients_dict = dict(gradients_counter)
        return gradients_dict


    def inplist2arr(self, inputlist):
        """
            This function covert a list of arrays to a single array by concatenating them.
        """
        for i in range(len(inputlist)):
            if i == 0:
                arr = inputlist[i]
            else:
                arr = np.concatenate((arr,inputlist[i]), axis=1)
        return arr

    def arr2inplist(self, arr):
        """
            This function covert a array to list of arrays based on num_samples.
        """
        inplist = [arr[:,0:1]]
        start_idx = 1
        layer_num = 1
        for i in self.num_samples:
            layer_num *= i
            end_idx = start_idx + layer_num
            inplist.append(arr[:,start_idx:end_idx])
            start_idx = end_idx
        return inplist
