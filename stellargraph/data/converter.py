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

from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.utils import to_categorical
from stellargraph.core.graph import StellarGraphBase


class NodeAttributeSpecification:
    """
    This class converts numeric and non-numeric node attributes to the appropriate
    numeric vectors for machine learning.

    In the StellarML library, all machine learning tasks that require feature and target
    attribute specifications should be passed an object of this class.

    # Usage

    Instantiation::

        nfs = NodeAttributeSpecification()

    To add an attribute for a node type, choose an appropriate Converter class
    and use the following methods:

    For a single attribute of node type node_type use `add_attribute`::

        nfs.add_attribute(node_type, attribute_name, Converter, <converter parameters>)

    For multiple attributes of a single node type, using a single converter class,
    use `add_attribute_list`::

        nfs.add_attribute_list(node_type, attribute_name, Converter, <converter paramters>)

    To add all attributes using the same converter class, use `add_all_attributes`,
    you will need to provide a StellarGraph object so that the node attributes can
    be extracted::

        nfs.add_all_attributes(graph, node_type, Converter, <converter paramters>)


    # Converter classes:

    There are multiple converter classes that can be used depending upon the
    attribute values and whether the attribute specification required is for features
    or targets.

    * BinaryConverter:
        This converter will create a value with a one if the attribute exists in the node
        attributes and zero if it does not.

    * CategorigalConverter:
        This converter takes an attribute that has multiple values (categories) and converts
        the categories to integers.

    * OneHotCategorigalConverter:
        This converter takes an attribute that has multiple values (categories) and converts
        the categories to one-hot vectors of length equal to the number of categories.

    * NumericConverter:
        This converter takes an attribute that has integer or floating point values and
        optionally normalizes them by mean and standard deviation.

    More inforamtion on these converters and their parameters can be found in their individual
    documentation. Also note that the converter parameters should be passed to the
    attribute specification methods, not directly to the converter.

    """

    def __init__(self):
        self._node_specs = {}
        self._node_feature_specs = {}

    def add_attribute(self, node_type, attr, converter, **conv_args):
        """
        Add a named attribute with specified converter for a node type

        Args:
            node_type: Node type that contains the attribute (must be specified, even
                if there is only a single node type)
            attr: Attribute name
            converter: Converter class (this should be the class, not an object)
            **conv_args: Optional arguemnts to the converter, specific to converter
        """

        if not issubclass(converter, StellarAttributeConverter):
            raise TypeError(
                "Converter should be a subclass of StellarAttributeConverter"
            )

        node_type_spec = self._node_specs.get(node_type, {})
        node_type_spec[attr] = converter(**conv_args)
        self._node_specs[node_type] = node_type_spec

    def add_attribute_list(self, node_type, attrs, converter, **conv_args):
        """
        Add multiple named attributes with the specified converter, note that
        an individual converter object will be created for each attribute.

        Args:
            node_type: Node type that contains the attribute names (must be specified,
                even if there is only a single node type)
            attrs: List of attribute names to use)
            converter: Converter class (this should be the class, not an object)
            **conv_args: Optional arguments to the converter, specific to converter
        """
        if not issubclass(converter, StellarAttributeConverter):
            raise TypeError(
                "Converter should be a subclass of StellarAttributeConverter"
            )

        node_type_spec = self._node_specs.get(node_type, {})
        for attr in attrs:
            node_type_spec[attr] = converter(**conv_args)
        self._node_specs[node_type] = node_type_spec

    def add_all_attributes(
        self, graph, node_type, converter, ignored_attributes=[], **conv_args
    ):
        """
        Add multiple named attributes with the specified converter to all
        attributes of the given node type found in the graph.

        Args:
            graph: A StellarGraph object containing nodes of the specified type.
            node_type: Node type that contains the attribute names (must be specified,
                even if there is only a single node type)
            converter: Converter class (this should be the class, not an object)
            ignored_attributes: (Optional) a list of attribute names to not include.
            **conv_args: Optional arguments to the converter, specific to converter
        """
        if not issubclass(converter, StellarAttributeConverter):
            raise TypeError(
                "Converter should be a subclass of StellarAttributeConverter"
            )
        if not isinstance(graph, StellarGraphBase):
            raise TypeError("Graph should be a StellarGraph")

        # Go through graph to find node attributes
        all_attrs = set(
            k for v in graph.nodes_of_type(node_type) for k in graph.node[v].keys()
        )

        # Remove any ignored attributes
        attrs = all_attrs.difference(set(ignored_attributes))

        # Don't use node type as attribute:
        attrs.discard(graph._node_type_attr)

        # Set found attributes with converter
        self.add_attribute_list(node_type, attrs, converter, **conv_args)

    def has_type(self, node_type):
        """
        Returns True if the specified type exists in the attribute specification

        Args:
            node_type: String specifying the node type

        Returns:
            A bool specifying if the node type exists.
        """
        return node_type in self._node_specs

    def get_types(self):
        """
        Returns a list of the node types in this attribute specification
        """
        return list(self._node_specs.keys())

    def get_attributes(self, node_type=None):
        """
        Get the list of attributes in a defined order for the given node type.

        Args:
            node_type: Node type key, if None and there is a single node type
                the attributes of that type are returned.

        Returns:
            List of attribute IDs
        """
        if node_type is None:
            if len(self._node_specs) == 1:
                node_attrs = next(iter(self._node_specs.values())).keys()
            else:
                raise RuntimeError(
                    "Please specify the node type when there are multiple node types"
                )

        elif node_type in self._node_specs:
            node_attrs = self._node_specs[node_type].keys()

        else:
            raise ValueError(
                "There are no nodes of type '{}' set as targets".format(node_type)
            )
        return sorted(node_attrs, key=str)

    def get_feature_indices(self, node_type):
        """
        Gives the ranges of the indices in the numeric vector
        corresponding to each attribute the specification.

        Args:
            node_type: The node type

        Returns:
            A dictionary of attribute index ranges in the form:
            ```
            { attribute_jj : (start_index, end_index) ... }
            ```
        """
        if node_type not in self._node_specs:
            return {}

        node_type_spec = self._node_specs[node_type]
        feature_list = sorted(node_type_spec.keys(), key=str)

        # Run over sorted array and map attribute to
        # range of values in the feature
        start_ind = 0
        feature_id_to_range = {}
        for attr in feature_list:
            conv = node_type_spec[attr]
            end_ind = start_ind + len(conv)
            feature_id_to_range[attr] = (start_ind, end_ind)
            start_ind = end_ind

        return feature_id_to_range

    def get_converter(self, node_type, attr):
        """
        Get the converter object for the specified node type and attribute name
        Args:
            node_type: Node type
            attr: Attribute name

        Returns:
            The converter object
        """

        if node_type not in self._node_specs:
            raise KeyError("Node type '{}' not in known node types.".format(node_type))
        if attr not in self._node_specs[node_type]:
            raise KeyError(
                "Attribute '{}' not known for node type {}.".format(attr, node_type)
            )
        return self._node_specs[node_type][attr]

    def get_output_size(self, node_type=None):
        """
        Get the size of the output vector for the node_type

        Args:
            node_type: The node type

        Returns:
            An integer specifying the vector length for this node type
        """
        if node_type is None:
            if len(self._node_specs) == 1:
                node_type = next(iter(self._node_specs.keys()))
            else:
                raise ValueError(
                    "Node type must be specified if there are multiple node types"
                )
        elif node_type not in self._node_specs:
            raise ValueError(
                "Node type '{}' not found in attribute specification.".format(node_type)
            )

        return np.sum([len(conv) for conv in self._node_specs[node_type].values()])

    def fit_transform(self, node_type, data):
        """
        Fit the converters for the given node type to the data and convert the
        data to output vectors.

        Args:
            node_type: The node type
            data: A list of dictionaries containing attribute names and values

        Returns:
            A numpy array containing the values of the converted attributes, of
            shape (length of data, output size)

        """
        n_data = len(data)

        # Convert attribute data to numeric values for each attribute
        converted_features = {}
        attr_list = self.get_attributes(node_type)
        for attr_name in attr_list:
            attr_data = [d.get(attr_name) for d in data]
            conv = self.get_converter(node_type, attr_name)
            converted_features[attr_name] = conv.fit_transform(attr_data)

        # Store features in array
        feature_array = np.concatenate(
            [
                np.reshape(converted_features[attr_name], (n_data, -1))
                for attr_name in attr_list
            ],
            axis=1,
        )
        return feature_array

    def transform(self, node_type, data):
        """
        Convert the supplied data to numeric vectors, this assumes that the converters
        have previously been trained.

        Args:
            node_type: The node type
            data: A list of dictionaries containing attribute names and values

        Returns:
            A numpy array containing the values of the converted attributes, of
            shape (length of data, output size)

        """
        n_data = len(data)

        # Convert attribute data to numeric values for each attribute
        converted_features = {}
        attr_list = self.get_attributes(node_type)
        for attr_name in attr_list:
            attr_data = [d.get(attr_name) for d in data]
            conv = self.get_converter(node_type, attr_name)
            converted_features[attr_name] = conv.transform(attr_data)

        # Store features in array
        feature_array = np.concatenate(
            [
                np.reshape(converted_features[attr_name], (n_data, -1))
                for attr_name in attr_list
            ],
            axis=1,
        )
        return feature_array

    def inverse_transform(self, node_type, data):
        """
        Convert the supplied numeric vectors back to the form of the original data.

        Args:
            node_type: The node type
            data: A numpy array of numeric data.

        Returns:
            A list containing the input attributes.

        """
        n_data = len(data)

        # The indices in the transformed vector for each attribute
        indices_for_attr = self.get_feature_indices(node_type)

        # Convert numeric values to the original domain for each attribute
        converted_features = {}
        attr_list = self.get_attributes(node_type)
        for attr_name in attr_list:
            conv = self.get_converter(node_type, attr_name)

            assert conv is not None
            assert attr_name in indices_for_attr

            # Extract data for this attribute
            index_range = indices_for_attr[attr_name]
            attr_data = data[:, index_range[0] : index_range[1]]

            converted_features[attr_name] = conv.inverse_transform(attr_data)

        # Convert to a list
        attr_out = [
            {attr_name: converted_features[attr_name][ii] for attr_name in attr_list}
            for ii in range(n_data)
        ]
        return attr_out


class StellarAttributeConverter(ABC):
    """
    Abstract class for attribute converters.
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass


class NumericConverter(StellarAttributeConverter):
    """
    This converter takes an attribute that has integer or floating point values and
    optionally normalizes them by mean and standard deviation.

    Args:
        dtype: (Optional) convert to a vector of this numpy data type
        default_value: (Optional) if the attribute is missing, if this is "mean" (default)
            assign the mean value calculated over the valid data, if this is a
            float or int, assign that value directly.
        normalize: (Optional) if this is "standard" normalize the values by shifting and
            scaling the values so that the mean is zero and the standard deviation is one.
    """

    def __init__(self, dtype="float32", default_value="mean", normalize="standard"):
        self.dtype = dtype
        self.normalize = normalize
        self.default_value = default_value

    def __len__(self):
        # TODO: extend this to multiple values
        return 1

    def fit_transform(self, data):
        data = np.asarray(data, dtype=self.dtype)

        # Calculate normalization parameters
        if self.normalize == "standard":
            self.scale = np.nanstd(data, axis=0)
            self.offset = np.nanmean(data, axis=0)
        else:
            self.scale = 1
            self.offset = 0

        if self.scale < 1e-6:
            raise ValueError(
                "When trying to normalize the data, the standard deviation close to zero."
            )

        return self.transform(data)

    def transform(self, data):
        data = np.asarray(data, dtype=self.dtype)

        # Normalization
        if self.normalize == "standard":
            data = (data - self.offset) / self.scale

        # Fill missing values
        if self.default_value == "mean":
            fill_value = np.nanmean(data)
        elif self.default_value == "median":
            fill_value = np.nanmedian(data)
        elif np.isscalar(self.default_value):
            fill_value = self.default_value

        data = np.where(np.isfinite(data), data, fill_value)
        return data

    def inverse_transform(self, data):
        data = np.asanyarray(data)

        # De-normalization
        if self.normalize == "standard":
            data = data * self.scale + self.offset

        # We can't un-fill missing values!
        return np.squeeze(data)


class CategoricalConverter(StellarAttributeConverter):
    """
    This converter takes an attribute that has multiple values (categories) and converts
    the categories to integers.

    Args:
        default_value: Value to assign to the vector output when the attribute is missing.
        dtype: (Optional) convert to a vector of this numpy data type

    """

    def __init__(self, default_value=0, dtype="float32"):
        self.default_value = default_value
        self.dtype = dtype
        self.categories = []

    def __len__(self):
        return 1

    def fit_transform(self, data):
        self.categories = sorted(set(data), key=str)
        return self.transform(data)

    def transform(self, data):
        # TODO: Checks for data input
        return np.array(
            [
                self.categories.index(d) if d is not None else self.default_value
                for d in data
            ],
            dtype=self.dtype,
        )

    def inverse_transform(self, data):
        # TODO: Checks for data input
        return [self.categories[int(ii)] for ii in data]


class OneHotCategoricalConverter(StellarAttributeConverter):
    """
    This converter takes an attribute that has multiple values (categories) and converts
    the categories to one-hot vectors of length equal to the number of categories.

    Args:
        default_value: (Optional) value to assign to the vector output when the attribute is missing.
        without_first: (Optional) Return a vector that omits the first value, so is zero when
            the first category is supplied. This can be useful for inputs to DL systems.
        dtype: (Optional) convert to a vector of this numpy data type
    """

    def __init__(self, default_value=0, without_first=False, dtype="float32"):
        self.default_value = default_value
        self.without_first = without_first
        self.dtype = dtype
        self.categories = []

    def fit_transform(self, data):
        self.categories = sorted(set(data), key=str)
        if len(self.categories) == 1:
            print("Warning: Only one category for attribute")

        return self.transform(data)

    def __len__(self):
        if self.without_first:
            size = len(self.categories) - 1
        else:
            size = len(self.categories)
        return size

    def transform(self, data):
        data_cats = [
            self.categories.index(d) if d is not None else self.default_value
            for d in data
        ]

        # Otherwise use the Keras to_categorical function
        data_trans = to_categorical(data_cats, len(self.categories)).astype(self.dtype)

        # If the without_first is set, remove the first value
        if self.without_first:
            data_trans = data_trans[:, 1:]

        return data_trans

    def inverse_transform(self, data):
        data = np.asanyarray(data)
        assert np.ndim(data) == 2

        # Get an integer category, adding one if we have without_first=True
        category_id = np.argmax(data, axis=1)
        if self.without_first:
            category_id = (category_id + 1) * np.any(data, axis=1).astype(int)

        return [self.categories[ii] for ii in category_id]


class BinaryConverter(StellarAttributeConverter):
    """
    This converter will create a value with a one if the attribute exists in the node
    attributes and zero if it does not.

    Args:
        default_value: Value to assign to the vector output when the attribute is missing.
        dtype: (Optional) convert to a vector of this numpy data type

    """

    def __init__(self, dtype="float32", default_value=0):
        self.dtype = dtype
        self.default_value = default_value

    def __len__(self):
        return 1

    def fit_transform(self, data):
        return self.transform(data)

    def transform(self, data):
        data_bool = [
            bool(d) if d is not None else bool(self.default_value) for d in data
        ]
        return np.asarray(data_bool, dtype=self.dtype)

    def inverse_transform(self, data):
        return [None if d == 0 else 1 for d in data]
