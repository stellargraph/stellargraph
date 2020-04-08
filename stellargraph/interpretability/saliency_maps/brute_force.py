# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from stellargraph.core.graph import StellarGraph
from stellargraph.core.element_data import ExternalIdIndex


__all__ = ["BruteForce", "node_importance"]


class _GraphWithNodeRemoved(StellarGraph):
    """
    A wrapper around StellarGraph to mimic a node being removed.

    Args:
        graph (StellarGraph): graph to remove a node from
        remove_node (any): node to remove
    """

    def __init__(self, graph, remove_node):
        # this doesn't call super().__init__() which should be fine since this doesn't need access
        # to any members of the superclass. It's only subclassing StellarGraph in order to avoid
        # failing the type checking done throughout the library.
        if not graph.has_node(remove_node):
            raise ValueError(
                f"remove_node: tried to remove '{remove_node}', but it doesn't exist in the provided graph."
            )
        self._graph = graph
        self._remove_node = remove_node

    def __getattr__(self, item):
        return getattr(self._graph, item)

    def node_features(self, nodes, node_type=None):
        features = self._graph.node_features(nodes=nodes, node_type=node_type)
        index = ExternalIdIndex(nodes)
        node_iloc = index.to_iloc([self._remove_node])

        if index.is_valid(node_iloc):
            features[node_iloc, :] = 0

        return features


class BruteForce:
    """
    Helper class to calculate brute-force importance scores.

    Args:
        graph (StellarGraph): Graph to predict on
        model (tensorflow.keras.Model): A trained Keras model
        create_generator (callable): Function that takes a graph and returns a generator object
            compatible with the model.
    """

    def __init__(self, graph, model, create_generator):
        self._model = model
        self._graph = graph
        self._create_generator = create_generator

    def node_importance(self, target_node, class_of_interest, nodes=None):
        """
        Calculate the importance of nodes on the prediction of the target node.

        Args:
            target_node (any): ID of target node
            class_of_interest (int): The index in the prediction array that the importance scores
                will be calculated for. If the model's prediction output is a single value, then
                this value should be set to zero.
            nodes (list of any, optional): IDs of nodes to calculate importance scores for. Defaults
                to using all nodes in the provided graph.

        Returns:
            Numpy array containing the node importance scores in the same order as the provided
            ``nodes`` parameter. If using all nodes, the scores are in the same order as
            ``graph.nodes()``.
        """
        return node_importance(
            self._graph,
            self._get_predict_func(target_node, class_of_interest),
            nodes=nodes,
        )

    def _sequence(self, graph, target_node):
        return self._create_generator(graph).flow(node_ids=[target_node])

    def _get_predict_func(self, target_node, class_of_interest):
        def predict(graph):
            prediction = np.squeeze(
                self._model.predict(self._sequence(graph, target_node))
            )
            if len(prediction.shape) == 0:
                if class_of_interest != 0:
                    raise ValueError(
                        f"class_of_interest: expected zero when prediction is a single value, but found: {class_of_interest}"
                    )
                return prediction
            elif len(prediction.shape) == 1:
                if class_of_interest >= prediction.size:
                    raise ValueError(
                        f"class_of_interest: expected to be less than the length of the prediction score array, "
                        f"but found 'class_of_interest': {class_of_interest}, length of prediction: {prediction.size}"
                    )
                return prediction[class_of_interest]
            else:
                raise ValueError(
                    f"expected model to produce a single row of prediction values, but found shape: {prediction.shape}"
                )

        return predict


def node_importance(graph: StellarGraph, predict, nodes=None):
    """
    Calculate importance of nodes for the prediction output.

    Args:
        graph (StellarGraph): Graph to predict on
        predict (callable): Function takes the graph as argument and returns a single numeric
            prediction value.
        nodes (list of any, optional): IDs of nodes to calculate importance scores for. Defaults to
            using all nodes in the provided graph.

    Returns:
        Numpy array containing the node importance scores in the same order as the provided
        ``nodes`` parameter. If using all nodes, the scores are in the same order as
        ``graph.nodes()``.
    """
    pred_baseline = predict(graph)

    if nodes is None:
        nodes = graph.nodes()

    def importance_of(node):
        perturbed = _GraphWithNodeRemoved(graph, remove_node=node)
        pred = predict(perturbed)
        return pred_baseline - pred

    return np.array([importance_of(node) for node in nodes])
