# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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
Mappers to provide input data for the graph models in layers.

"""
__all__ = [
    "NodeSequence",
    "GraphSAGENodeGenerator",
    "HinSAGENodeGenerator",
    "FullBatchNodeGenerator",
    "FullBatchNodeSequence",
]

import warnings
import operator
import random
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
import keras.backend as K
from functools import reduce
from keras.utils import Sequence

from ..data.explorer import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
)
from ..core.graph import StellarGraphBase, GraphSchema
from ..core.utils import is_real_iterable


class NodeSequence(Sequence):
    """Keras-compatible data generator to use with the Keras
    methods :meth:`keras.Model.fit_generator`, :meth:`keras.Model.evaluate_generator`,
    and :meth:`keras.Model.predict_generator`.

    This class generated data samples for node inference models
    and should be created using the `.flow(...)` method of
    :class:`GraphSAGENodeGenerator` or :class:`HinSAGENodeGenerator`.

    These Generators are classes that capture the graph structure
    and the feature vectors of each node. These generator classes
    are used within the NodeSequence to generate samples of k-hop
    neighbourhoods in the graph and to return to this class the
    features from the sampled neighbourhoods.

    Args:
        generator: GraphSAGENodeGenerator or HinSAGENodeGenerator
            The generator object containing the graph information.
        ids: list
            A list of the node_ids to be used as head-nodes in the
            downstream task.
        targets: list, optional (default=None)
            A list of targets or labels to be used in the downstream
            class.

        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.

    """

    def __init__(self, generator, ids, targets=None, shuffle=True):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids"
                )
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Check all IDs are actually in the graph
        if any(n not in generator.graph for n in ids):
            raise KeyError(
                "Head nodes supplied to generator contain IDs not found in graph"
            )

        # Infer head_node_type
        if (
            generator.schema.node_type_map is None
            or generator.schema.edge_type_map is None
        ):
            raise RuntimeError("Schema must have node and edge type maps.")
        else:
            head_node_types = {generator.schema.get_node_type(n) for n in ids}
        if len(head_node_types) > 1:
            raise ValueError(
                "Only a single head node type is currently supported for HinSAGE models"
            )
        head_node_type = head_node_types.pop()

        # Store the generator to draw samples from graph
        self.generator = generator
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle

        # Shuffle IDs to start
        self.on_epoch_end()

        # Save head node type and generate sampling schema
        self.head_node_types = [head_node_type]
        self._sampling_schema = generator.schema.sampling_layout(
            self.head_node_types, generator.num_samples
        )

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.generator.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data

        Args:
            batch_num (int): number of a batch

        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.

        """
        start_idx = self.generator.batch_size * batch_num
        end_idx = start_idx + self.generator.batch_size
        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching batch {} [{}]".format(batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get corresponding targets
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get sampled nodes
        batch_feats = self.generator.sample_features(head_ids, self._sampling_schema)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all head (root) nodes at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            random.shuffle(self.indices)


class GraphSAGENodeGenerator:
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        schema (GraphSchema): [Optional] Graph schema for G.
        seed (int): [Optional] Random seed for the node sampler.
        name (str or None): Name of the generator (optional)
    """

    def __init__(self, G, batch_size, num_samples, schema=None, seed=None, name=None):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # Check if the graph has features
        G.check_graph_for_ml()

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema(create_type_maps=True)
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # Check that there is only a single node type for GraphSAGE
        if len(self.schema.node_types) > 1:
            print(
                "Warning: running homogeneous GraphSAGE on a graph with multiple node types"
            )

        # Create sampler for GraphSAGE
        self.sampler = SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=seed)

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            sampling_schema: The sampling schema for the model

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        # The number of samples for each head node (not including itself)
        num_full_samples = np.sum(np.cumprod(self.num_samples))

        # Isolated nodes will return only themselves in the sample list
        # let's correct for this by padding with None (the dummy node ID)
        node_samples = [
            ns + [None] * num_full_samples if len(ns) == 1 else ns
            for ns in node_samples
        ]

        # Reshape node samples to sensible format
        def get_levels(loc, lsize, samples_per_hop, walks):
            end_loc = loc + lsize
            walks_at_level = list(it.chain(*[w[loc:end_loc] for w in walks]))
            if len(samples_per_hop) < 1:
                return [walks_at_level]
            return [walks_at_level] + get_levels(
                end_loc, lsize * samples_per_hop[0], samples_per_hop[1:], walks
            )

        nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)
        node_type = sampling_schema[0][0][0]

        # Get features for sampled nodes
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, node_type)
            for layer_nodes in nodes_per_hop
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]
        return batch_feats

    def flow(self, node_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                `(len(node_ids), target_size)`
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(self, node_ids, targets, shuffle=shuffle)

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods ``fit_generator``, ``evaluate_generator``,
            and ``predict_generator``

        """
        return NodeSequence(
            self, node_targets.index, node_targets.values, shuffle=shuffle
        )


class HinSAGENodeGenerator:
    """Keras-compatible data mapper for Heterogeneous GraphSAGE (HinSAGE)

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the HinSAGE model.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features for all
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Note that the shuffle argument should be True for training and
    False for prediction.

     Example::

         G_generator = HinSAGENodeGenerator(G, 50, [10,10])
         train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
         test_data_gen = G_generator.flow(test_node_ids)

     """

    def __init__(self, G, batch_size, num_samples, schema=None, seed=None, name=None):
        """

        Args:
            G (StellarGraph): The machine-learning ready graph
            batch_size (int): Size of batch to return
            num_samples (list): The number of samples per layer (hop) to take
            schema (GraphSchema): [Optional] Graph schema for G.
            seed (int), Optional: Random seed for the node sampler
            name (str), optional: Name of the generator.
        """
        self.graph = G
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.name = name

        # We require a StellarGraph
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        G.check_graph_for_ml(features=True)

        # Generate schema
        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema(create_type_maps=True)
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def sample_features(self, head_nodes, sampling_schema):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            sampling_schema: The node sampling schema for the HinSAGE model,
                this is can be generated by the ``GraphSchema`` object.

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where num_sampled_at_layer is the cumulative product of `num_samples`
            for that layer.
        """
        # Get sampled nodes
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        # Reshape node samples to the required format for the HinSAGE model
        # This requires grouping the sampled nodes by edge type and in order
        nodes_by_type = [
            (
                nt,
                reduce(
                    operator.concat,
                    (samples[ks] for samples in node_samples for ks in indices),
                    [],
                ),
            )
            for nt, indices in sampling_schema[0]
        ]

        # Get features
        batch_feats = [
            self.graph.get_feature_for_nodes(layer_nodes, nt)
            for nt, layer_nodes in nodes_by_type
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]

        return batch_feats

    def flow(self, node_ids, targets=None, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downsteam task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids (iterable): The head node IDs
            targets (Numpy array): a 2D array of numeric targets with shape
                ``(len(node_ids), target_size)``
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods `fit_generator`, `evaluate_generator`,
            and `predict_generator`.

        """
        return NodeSequence(self, node_ids, targets, shuffle=shuffle)

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_targets (DataFrame): Numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with the GraphSAGE model
            in Keras methods `fit_generator`, `evaluate_generator`,
            and `predict_generator`.
        """
        return NodeSequence(
            self, node_targets.index, node_targets.values, shuffle=shuffle
        )


class SparseFullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit_generator`,
        :meth:`keras.Model.evaluate_generator`, and
        :meth:`keras.Model.predict_generator`,

    This class uses sparse matrix representations to send data to the models,
    and only works with the Keras tensorflow backend. For any other backends,
    use the :class:`FullBatchNodeSequence` class.

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (np.ndarray or sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = True

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Store features and targets as np.ndarray
        self.features = np.asanyarray(features)
        self.target_indices = np.asanyarray(indices)

        # Ensure matrix is in COO format to extract indices
        if sps.isspmatrix(A):
            A = A.tocoo()

        else:
            raise ValueError("Adjacency matrix not in expected sparse format")

        # Convert matrices to list of indices & values
        self.A_indices = np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])), 0)
        self.A_values = np.expand_dims(A.data, 0)

        # Reshape all inputs to have batch dimension of 1
        self.target_indices = np.reshape(
            self.target_indices, (1,) + self.target_indices.shape
        )
        self.features = np.reshape(self.features, (1,) + self.features.shape)
        self.inputs = [
            self.features,
            self.target_indices,
            self.A_indices,
            self.A_values,
        ]

        if targets is not None:
            self.targets = np.asanyarray(targets)
            self.targets = np.reshape(self.targets, (1,) + self.targets.shape)
        else:
            self.targets = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class FullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models
    that require full-batch training (e.g., GCN, GAT).
    Use this class with the Keras methods :meth:`keras.Model.fit_generator`,
        :meth:`keras.Model.evaluate_generator`, and
        :meth:`keras.Model.predict_generator`,

    This class should be created using the `.flow(...)` method of
    :class:`FullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        A (np.ndarray or sparse matrix): An adjacency matrix of the graph of size (N x N).
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    use_sparse = False

    def __init__(self, features, A, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        # Store features and targets as np.ndarray
        self.features = np.asanyarray(features)
        self.target_indices = np.asanyarray(indices)

        # Convert sparse matrix to dense:
        if sps.issparse(A) and hasattr(A, "toarray"):
            self.A_dense = A.toarray()
        elif isinstance(A, (np.ndarray, np.matrix)):
            self.A_dense = np.asanyarray(A)
        else:
            raise TypeError(
                "Expected input matrix to be either a Scipy sparse matrix or a Numpy array."
            )

        # Reshape all inputs to have batch dimension of 1
        self.features = np.reshape(self.features, (1,) + self.features.shape)
        self.A_dense = self.A_dense.reshape((1,) + self.A_dense.shape)
        self.target_indices = np.reshape(
            self.target_indices, (1,) + self.target_indices.shape
        )

        self.inputs = [self.features, self.target_indices, self.A_dense]

        if targets is not None:
            self.targets = np.asanyarray(targets)
            self.targets = np.reshape(self.targets, (1,) + self.targets.shape)
        else:
            self.targets = None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets


class FullBatchNodeGenerator:
    """
    A data generator for node prediction with Homogeneous full-batch models, e.g., GCN, GAT, SGC.
    The supplied graph G should be a StellarGraph object that is ready for
    machine learning. Currently the model requires node features to be available for all
    nodes in the graph.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = FullBatchNodeGenerator(G)
        train_data_gen = G_generator.flow(node_ids, node_targets)

        # Fetch the data from train_data_gen, and feed into a Keras model:
        [X, A], y_train, node_mask_train = train_data_gen.__getitem__(0)
        model.fit(x=[X, A], y=y_train, sample_weight=node_mask_train, ...)

        # Alternatively, use the generator itself with model.fit_generator:
        model.fit_generator(train_gen, epochs=num_epochs, ...)

    Args:
        G (StellarGraphBase): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        func_opt: an optional function to apply on features and adjacency matrix
            (declared func_opt(features, Aadj, **kwargs))
        k (None or int): If not none and filter is smoothed, the normalised adjacency matrix with self loops will be
            raised to the k-th power before multiplying by the node features.
        kwargs: additional parameters needed when using this generator with GCN model with the [func_opt] function.
            It must be chebyshev, localpool, or smoothed filters (e.g. filter="localpool", or
            filter="chebyshev", max_degree=2, or filter="smoothed"). For more information, please read
            `GCN_Aadj_feats_op <https://github.com/stellargraph/stellargraph/tree/master/stellargraph/core>`_ in the
            file **utils.py** and GCN demo
            `gcn-cora-example.py <https://github.com/stellargraph/stellargraph/blob/master/demos/node-classification-gcn/gcn-cora-example.py>`_
    """

    def __init__(self, G, name=None, func_opt=None, k=None, **kwargs):
        if not isinstance(G, StellarGraphBase):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.name = name
        self.k = k
        self.kwargs = kwargs
        self.number_of_nodes = G.number_of_nodes()

        # Check if the graph has features
        G.check_graph_for_ml()

        # Create sparse adjacency matrix
        self.node_list = list(G.nodes())
        self.Aadj = nx.to_scipy_sparse_matrix(
            G, nodelist=self.node_list, dtype="float32", weight="weight", format="coo"
        )

        # Power-user feature: make the generator yield dense adjacency matrix instead
        # of the default sparse one.
        self.use_sparse = kwargs.get("sparse", True)

        # If sparse is specified, check that the backend is tensorflow
        if self.use_sparse and K.backend() != "tensorflow":
            warnings.warn(
                "Sparse adjacency matrices are only supported in tensorflow."
                " Falling back to using a dense adjacency matrix."
            )

        # We need a schema to check compatibility with GAT, GCN
        self.schema = G.create_graph_schema(create_type_maps=True)

        # Check that there is only a single node type for GAT or GCN
        if len(self.schema.node_types) > 1:
            raise TypeError(
                "{}: node generator requires graph with single node type; "
                "a graph with multiple node types is passed. Stopping.".format(
                    type(self).__name__
                )
            )

        # Get the features for the nodes
        self.features = G.get_feature_for_nodes(self.node_list)

        if func_opt is not None:
            if callable(func_opt):
                _, self.Aadj = func_opt(
                    features=self.features, A=self.Aadj, k=self.k, **kwargs
                )
            else:
                raise ValueError("argument 'func_opt' must be callable")

    def flow(self, node_ids, targets=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: and iterable of node ids for the nodes of interest (e.g., training, validation, or test set nodes)
            targets: a 2D array of numeric node targets with shape `(len(node_ids), target_size)`

        Returns:
            A NodeSequence object to use with GCN or GAT models
            in Keras methods :meth:`fit_generator`, :meth:`evaluate_generator`,
            and :meth:`predict_generator`

        """
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")

        # The list of indices of the target nodes in self.node_list
        node_indices = np.array([self.node_list.index(n) for n in node_ids])

        if self.use_sparse:
            return SparseFullBatchNodeSequence(
                self.features, self.Aadj, targets, node_indices
            )
        else:
            return FullBatchNodeSequence(
                self.features, self.Aadj, targets, node_indices
            )
