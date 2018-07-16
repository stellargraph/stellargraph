"""
Link Attribute Inference on a Heterogeneous Graph,

Containing functions for the MovieLens dataset (Move these elsewhere?)
"""
import time
import os
import argparse
import pickle
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import functools as ft

from collections import defaultdict
from typing import AnyStr, Any, List, Tuple, Dict, Optional, Callable

from keras import backend as K
from keras import Input, Model, optimizers, losses, activations, metrics
from keras.layers import Layer, Dense, Concatenate, Multiply, Activation, Lambda, Reshape
from keras.utils import Sequence

from stellar.layer.hinsage import Hinsage


class LeakyClippedLinear(Layer):
    """Leaky Clipped Linear Unit.

        Args:
            low (float): Lower threshold
            high (float): Lower threshold
            alpha (float) The slope of the function below low or above high.
    """
    def __init__(self, low: float=1.0, high: float=5.0, alpha: float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.gamma = K.cast_to_floatx(1 - alpha)
        self.lo = K.cast_to_floatx(low)
        self.hi = K.cast_to_floatx(high)

    def call(self, x, mask=None):
        x_lo = K.relu(self.lo - x)
        x_hi = K.relu(x - self.hi)
        return x + self.gamma * x_lo - self.gamma * x_hi

    def get_config(self):
        config = {
            'alpha': float(1 - self.gamma),
            'low': float(self.lo),
            'high': float(self.hi)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class EdgeDataGenerator(Sequence):
    'Geras data generator for edge prediction'

    def __init__(self,
                 data: Any,
                 ids: List[Tuple[int, int]],
                 labels: np.ndarray,
                 samples_per_hop: List[int],
                 batch_size: int = 1000,
                 name: AnyStr = 'train'):
        """Generate Data for the edge inference problem

        Args:
            ids (List[Tuple[int, int]]): Edge IDs -> Tuple of (src, dst)
            labels (List[int]): Labels corresponding to previous edge IDs
            number_of_samples (int):
            batch_size (int, optional): Defaults to 1000.
            name (AnyStr, optional): Defaults to 'train'.
        """
        self.data = data
        self.batch_size = batch_size
        self.ids = ids
        self.labels = labels
        self.data_size = len(ids)
        self.samples_per_hop = samples_per_hop
        self.name = name
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num: int):
        'Generate one batch of data'
        start_idx = self.batch_size * batch_num
        if (start_idx >= self.data_size):
            print("this shouldn't happen ... ever")
            start_idx = 0
        end_idx = start_idx + self.batch_size

        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get edges and labels
        edges = self.ids[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # Extract head nodes from edges
        head_size = len(edges)
        head_nodes = [[e[ii] for e in edges] for ii in range(2)]

        # Features from head nodes
        batch_feats = [self.data.features[hn] for hn in head_nodes]

        # Sample subgraph of head nodes and collect features
        for num_for_hop in self.samples_per_hop:
            for ii in range(2):
                indices = self.data.sample_neighbours(head_nodes[ii],
                                                      num_for_hop)
                batch_feats.append(self.data.features[indices])
                head_nodes[ii] = indices

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            f.reshape(head_size, -1, self.data.input_feature_size)
            for f in batch_feats
        ]

        return batch_feats, batch_labels


class MovielensData:
    """
    Class to manage movielens graph data.

    Args:
        graph_loc: Location of networkx graph file
        features_loc: Location of features file
        target_name: Name of edge target attribute to predict.
    """

    def __init__(self, graph_loc: AnyStr, features_loc: AnyStr,
                 target_name: AnyStr):
        # Read graph
        print("Reading graph...")
        G = nx.read_gpickle(graph_loc)

        # Read features
        print("Reading features...")
        with open(features_loc, 'rb') as f:
            self.features = pickle.load(f)

        # Feature size
        self.input_feature_size: int = self.features.shape[1]

        # number of samples per additional layer,
        self.node_samples: List[int] = [30, 10]

        # number of features per additional layer
        self.layer_sizes: List[int] = [50, 50]

        # Per-node baselines - learns a baseline for movies and users
        # requires fixed set of train/test movies + users
        self.node_baseline: bool = True

        # Layer bias
        self.use_bias: bool = True

        # Dropout
        self.dropout: float = 0.0

        # Specify heterogeneous sampling schema
        self.subtree_schema = [
            ('user', [2]),
            ('movie', [3]),
            ('movie', [4]),
            ('user', [5]),
            ('user', []),
            ('movie', []),
        ]

        # Training and test edges
        train_edges = [e for e in G.edges if G.edges[e]['split'] == 0]
        test_edges = [e for e in G.edges if G.edges[e]['split'] == 1]

        # Adjacency lists
        self.adj_list_train: Dict[int, List[int]] = self.nx_to_adj(
            G, edge_splits=[0])
        self.adj_list_test: Dict[int, List[int]] = self.nx_to_adj(
            G, edge_splits=[0, 1])

        # Directed edgelist
        # Note that this assumes that the movie IDs are lower than the user IDs
        self.edgelist_train = [(min(e), max(e)) for e in train_edges]
        self.edgelist_test = [(min(e), max(e)) for e in test_edges]

        # Create list of target labels
        self.labels_train = np.array(
            [G.edges[e][target_name] for e in train_edges])
        self.labels_test = np.array(
            [G.edges[e][target_name] for e in test_edges])

    def shapes_from_schema(self) -> List[Tuple[int, int]]:
        """This comes from the schema directly

        Returns:
         [
            Input(shape=shape_at(0)),
            Input(shape=shape_at(0)),
            Input(shape=shape_at(1)),
            Input(shape=shape_at(1)),
            Input(shape=shape_at(2)),
            Input(shape=shape_at(2))
        ]
        """

        def shape_at(i: int) -> Tuple[int, int]:
            return (np.product(self.node_samples[:i], dtype=int),
                    self.input_feature_size)

        def from_schema(schema: List[Tuple[AnyStr, List[int]]]
                        ) -> List[Tuple[int, int]]:
            out_shapes = []
            level_indices = defaultdict(lambda: 0)
            for ii, s in enumerate(schema):
                level = level_indices[ii]
                for c in s[1]:
                    level_indices[c] = level + 1
                out_shapes.append(shape_at(level))
            return out_shapes

        return from_schema(self.subtree_schema)

    def nx_to_adj(self, G: nx.Graph,
                  edge_splits: List[int]) -> Dict[int, List[int]]:
        """Convert networkx graph to adjacency list

        Args:
            G (nx.Graph): NetworkX graph
            edge_split (List[int]): Edge property to filter on.

        Returns:
            [type]: Dict
        """
        adj = defaultdict(lambda: [-1])
        for n, nbrdict in G.adjacency():
            neighs = [
                k for k, edge_info in nbrdict.items()
                if edge_info['split'] in edge_splits
            ]
            # If no edges in selected split, continue
            if len(neighs) == 0:
                continue
            adj[n] = neighs
        return adj

    def sample_neighbours(self,
                          indices: List[int],
                          ns: int,
                          train: bool = True) -> List[List[int]]:
        """Returns ns nodes sampled from the neighbours of the supplied indices.

        Args:
            indices (List[int]): List of nodes to sample neigbhours of
            ns (int): Number of sampled neighbours, per node
            train (bool, optional): Use training graph if True, else test graph.

        Returns:
            List[List[int]]: List of neighbour samples for each index.
        """

        adj = self.adj_list_train if train else self.adj_list_test
        return [
            adj[inx][i] for inx in indices
            for i in np.random.randint(len(adj[inx]), size=ns)
        ]

    def create_model(self) -> Hinsage:
        return Hinsage(
            output_dims=self.layer_sizes,
            n_samples=self.node_samples,
            input_neigh_tree=self.subtree_schema,
            input_dim={
                'user': self.input_feature_size,
                'movie': self.input_feature_size,
            },
            bias=self.use_bias,
            dropout=self.dropout)

    def edge_data_generators(self, batch_size: int) -> Tuple[Sequence]:
        train_gen = EdgeDataGenerator(
            data=self,
            ids=self.edgelist_train,
            labels=self.labels_train,
            samples_per_hop=self.node_samples,
            batch_size=batch_size,
            name='train')

        test_gen = EdgeDataGenerator(
            data=self,
            ids=self.edgelist_test,
            labels=self.labels_test,
            samples_per_hop=self.node_samples,
            batch_size=batch_size,
            name='test')
        return train_gen, test_gen


def classification_predictor(hidden_1: Optional[int] = None,
                             hidden_2: Optional[int] = None,
                             output_act: AnyStr = 'softmax',
                             method: AnyStr = 'ip'):
    """Returns a function that predicts an edge classification output from node features.

        hidden_1 ([type], optional): Hidden size for the transform of node 1 features.
        hidden_2 ([type], optional): Hidden size for the transform of node 1 features.
        edge_function (str, optional): One of 'ip', 'mul', and 'concat'

    Returns:
        Function taking HinSAGE edge tensors and returning a logit function.
    """
    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if hidden_1:
            x0 = Dense(hidden_1, activation='relu')(x0)

        if hidden_2:
            x1 = Dense(hidden_2, activation='relu')(x1)

        if method == 'ip':
            out = Lambda(
                lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                    [x0, x1])

        elif method == 'mul':
            le = Multiply()([x0, x1])
            out = Dense(2, activation=output_act)(le)

        elif method == 'concat':
            le = Concatenate()([x0, x1])
            out = Dense(2, activation=output_act)(le)

        return out

    return edge_function


def regression_predictor(hidden_1: Optional[int] = None,
                         hidden_2: Optional[int] = None,
                         clip_limits: Optional[Tuple[int]] = None,
                         method: AnyStr = 'ip'):
    """Returns a function that predicts an edge regression output from node features.

        hidden_1 ([type], optional): Hidden size for the transform of node 1 features.
        hidden_2 ([type], optional): Hidden size for the transform of node 1 features.
        edge_function (str, optional): One of 'ip', 'mul', and 'concat'

    Returns:
        Function taking HinSAGE edge tensors.
    """

    def edge_function(x):
        x0 = x[0]
        x1 = x[1]

        if hidden_1:
            x0 = Dense(hidden_1, activation='relu')(x0)

        if hidden_2:
            x1 = Dense(hidden_2, activation='relu')(x1)

        if method == 'ip':
            out = Lambda(
                lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
                    [x0, x1])

        elif method == 'mul':
            le = Multiply()([x0, x1])
            out = Dense(1, activation='linear')(le)
            out = Reshape((1,))(out)

        elif method == 'concat':
            le = Concatenate()([x0, x1])
            out = Dense(1, activation='linear')(le)
            out = Reshape((1,))(out)

        if clip_limits:
            out = LeakyClippedLinear(
                low=clip_limits[0], high=clip_limits[1], alpha=0.1)(out)
        return out

    return edge_function


def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))
