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
import itertools as it
import numpy as np
from keras.utils.np_utils import to_categorical


class NodeFeatureConverter:
    def __init__(
        self,
        from_graph,
        to_graph=None,
        ignored_attributes=[],
        attributes=None,
        remove_converted_attrs=False,
        dtype="float32",
    ):
        # Form consistent node indexing
        self.node_list = sorted(from_graph, key=str)
        n_nodes = len(self.node_list)

        # Inverse ID to index mapping:
        self.node_id_to_index = {v: ii for ii, v in enumerate(self.node_list)}

        # Get a set of attributes to use for features
        if attributes is None:
            all_attrs = set(
                it.chain(*[from_graph.nodes[v].keys() for v in self.node_list])
            )
        else:
            all_attrs = set(attributes)

        # Remove any ignored attributes
        feature_attr = all_attrs.difference(set(ignored_attributes))

        # sets are unordered, so sort them to ensure reproducible order:
        self.feature_list = sorted(feature_attr)
        self.feature_id_to_index = {a: ii for ii, a in enumerate(feature_attr)}

        # Feature size will be number of found attributes
        self.feature_size = len(self.feature_list)

        # Store features in graph
        if to_graph is not None:
            # Set the "feature" and encoded "target" attributes for all nodes in the graph.
            for v in self.node_list:
                vdata = from_graph.nodes[v]

                # Decode attributes to a feature array
                node_feature_array = np.zeros(self.feature_size, dtype=dtype)
                for attr_name, attr_value in vdata.items():
                    index = self.feature_id_to_index.get(attr_name)
                    if index:
                        node_feature_array[index] = attr_value

                # Store feature array in graph
                to_graph.nodes[v]["feature"] = node_feature_array

                # Remove attributes
                if remove_converted_attrs:
                    for attr_name in vdata.keys():
                        if attr_name in feature_attr:
                            del vdata[attr_name]

        # Store features in indexed array
        else:
            self.feature_array = np.zeros((n_nodes, self.feature_size), dtype=dtype)

            # Set the "feature" and encoded "target" attributes for all nodes in the graph.
            for ii, v in enumerate(self.node_list):
                vdata = from_graph.nodes[v]

                # Decode attributes to a feature array
                for attr_name, attr_value in vdata.items():
                    index = self.feature_id_to_index.get(attr_name)
                    if index:
                        self.feature_array[ii, index] = attr_value

                # Remove converted attributes
                if remove_converted_attrs:
                    for attr_name in vdata.keys():
                        if attr_name in feature_attr:
                            del vdata[attr_name]

    def __call__(self, id=None, index=None):
        if id is not None:
            index = self.feature_id_to_index.get(id)

        elif index is None:
            raise ValueError(
                "{} must be given a node ID or index".format(type(self).__name__)
            )

        if index is None:
            raise ValueError("Couldn't find feature for node ID {}".format(id))

        return self.feature_array[index]

    def __len__(self):
        return self.feature_size


class NodeTargetConverter:
    """
    Targets need to be converted to a numeric value, if not one already.
    Depening on the downstream model, different conversions are required.

    * If target_type is None, convert the target attributes to float
      (e.g. for regression)

    * If target_type is a numpy dtype, convert the target attributes that dtype
      (e.g. for regression or binary classification)

    * If the target is 'categorical' encode the target categories
      as integers between 0 and number_of_categories - 1

    * If the target is '1hot' or 'onehot' encode the target categories
      as a binary vector of length number_of_categories.
      see the Keras function keras.utils.np_utils.to_categorical

    """

    def __init__(self, from_graph, target=None, target_type=None):
        self._graph = from_graph
        self._target_attr = target

        # THis is none for non-categorical values
        self.target_category_values = None

        if target_type is None:
            self.target_to_value = lambda x: np.float32(x)
            self.value_to_target = lambda x: np.float32(x)

        elif target_type == "categorical":
            self.target_category_values = sorted(
                set([from_graph.node[n][target] for n in from_graph.nodes()])
            )

            self.target_to_value = lambda x: self.target_category_values.index(x)
            self.value_to_target = lambda x: self.target_category_values[x]

        elif target_type == "1hot" or target_type == "onehot":
            self.target_category_values = sorted(
                set([from_graph.node[n][target] for n in from_graph.nodes()])
            )
            self.value_to_target = lambda x: self.target_category_values[
                np.argmax(x, axis=1)
            ]
            self.target_to_value = lambda x: to_categorical(
                self.target_category_values.index(x), len(self.target_category_values)
            )

        elif target_type in np.typeDict:
            dtype = np.typeDict[target_type]
            self.target_to_value = lambda x: dtype(x)
            self.value_to_target = lambda x: dtype(x)

        else:
            raise ValueError("Target type '{}' is not supported.".format(target_type))

    def __len__(self):
        if self.target_category_values:
            return len(self.target_category_values)
        else:
            return None

    def __call__(self, x, inverse=False):
        if not inverse:
            return self.target_to_value(x)
        else:
            return self.value_to_target(x)

    def convert_node_label_pairs(self, node_pairs_list):
        """
        Convert a list of (node_id, label_id) pairs to a list of
        node_ids and label values.

        Args:
            node_pairs_list: List of (node_id, target_id) pairs.

        Returns:
            A list of node_ids and a list of target values to
            be passed to a mapper object.
        """
        ids = [v[0] for v in node_pairs_list]
        label_values = np.array([self(v[1]) for v in node_pairs_list])
        return ids, label_values

    def get_targets_for_ids(self, node_ids):
        """
        Convert a list of node IDs to a list of
        node_ids and label values.

        Args:
            node_ids: List of node IDs in the graph.

        Returns:
            A list of node_ids and a list of target values to
            be passed to a mapper object.
        """
        label_values = np.array(
            [self(self._graph.nodes[v][self._target_attr]) for v in node_ids]
        )
        return node_ids, label_values

    def get_node_label_pairs(self, node_ids=None, convert=False):
        """
        Get a list of node IDs and labels for the splitter.

        Args:
            node_ids: List of node IDs in the graph, if not specified all
                nodes will be used.
            convert: If True the raw target values will be converted
                to the required numeric representation,
                if False they will be returned as raw values.

        Returns:
            A list tuples of (node_id, target) to be passed to the splitter.
        """
        if node_ids is None:
            node_ids = list(self._graph)

        if convert:
            id_target_pairs = [
                (v, self(self._graph.nodes[v][self._target_attr])) for v in node_ids
            ]
        else:
            id_target_pairs = [
                (v, self._graph.nodes[v][self._target_attr]) for v in node_ids
            ]

        return id_target_pairs
