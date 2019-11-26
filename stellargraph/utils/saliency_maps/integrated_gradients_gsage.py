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
from .saliency_gsage import GradientSaliencyGSAGE
import scipy.sparse as sp
from tensorflow.keras import backend as K


class IntegratedGradientsGSAGE(GradientSaliencyGSAGE):
    """
    A SaliencyMask class that implements the integrated gradients method.
    """

    def __init__(self, model, generator):
        super().__init__(model, generator)

    def get_integrated_node_masks(
        self,
        node_idx,
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
        X_val = self.X
        X_val = self.inplist2arr(X_val)
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
            X_step = self.arr2inplist(X_step)
            total_gradients += super().get_node_masks(
                node_idx, class_of_interest, X_val=X_step
            )

        return total_gradients / steps * X_diff

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
        A_val = self.A
        A_val = self.inplist2arr(A_val)
        total_gradients = np.zeros(A_val.shape)
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            tmp = super().get_link_masks(
                alpha, class_of_interest, int(non_exist_edge)
            )
            total_gradients += tmp
        link_gradients_list = self.arr2inplist(total_gradients / steps)
        link_gradients_dict = self.link_gradlist2dict(link_gradients_list, self.L)
        return link_gradients_dict

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
        node_grads = np.sum(gradients,axis=2)
        gradlist = self.arr2inplist(node_grads)
        gradients_dict = self.node_gradlist2dict(gradlist, self.A)
        return gradients_dict

    def expected_node_importance(self, node_id, class_of_interest, ig_steps=20, exp_steps=20):
        from collections import Counter
        node_importance_counter = Counter()
        for i in range(exp_steps):
            self.batch_gen()
            gradients_dict = self.get_node_importance(node_id, class_of_interest, steps=ig_steps)
            node_importance_counter.update(Counter(gradients_dict))
        for i in node_importance_counter.keys():
            node_importance_counter[i] /= exp_steps
        return dict(node_importance_counter)

    def expected_link_importance(self, node_id, class_of_interest, ig_steps=20, exp_steps=20):
        from collections import Counter
        link_importance_counter = Counter()
        for i in range(exp_steps):
            self.batch_gen()
            gradients_dict = self.get_link_importance(node_id, class_of_interest, steps=ig_steps)
            link_importance_counter.update(Counter(gradients_dict))
        for i in link_importance_counter.keys():
            link_importance_counter[i] /= exp_steps
        return dict(link_importance_counter)

