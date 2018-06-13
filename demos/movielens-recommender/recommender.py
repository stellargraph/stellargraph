"""
Link Attribute Inference on a Heterogeneous Graph for the MovieLens dataset

usage: movielens.py [-h] [-c [CHECKPOINT]] [-n BATCH_SIZE] [-e EPOCHS]
                    [-s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]]
                    [-l [LAYER_SIZE [LAYER_SIZE ...]]] [-m METHOD] [-b]
                    [-g GRAPH] [-f FEATURES] [-t TARGET]

Run GraphSAGE on movielens

optional arguments:
  -h, --help            show this help message and exit
  -c [CHECKPOINT], --checkpoint [CHECKPOINT]
                        Load a save checkpoint file
  -n BATCH_SIZE, --batch_size BATCH_SIZE
                        Load a save checkpoint file
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for
  -s [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]], --neighbour_samples [NEIGHBOUR_SAMPLES [NEIGHBOUR_SAMPLES ...]]
                        The number of nodes sampled at each layer
  -l [LAYER_SIZE [LAYER_SIZE ...]], --layer_size [LAYER_SIZE [LAYER_SIZE ...]]
                        The number of hidden features at each layer
  -m METHOD, --method METHOD
                        The edge regression method: 'concat', 'mul', or 'ip
  -b, --baseline        Use a learned offset for each node.
  -g GRAPH, --graph GRAPH
                        The graph stored in networkx pickle format.
  -f FEATURES, --features FEATURES
                        The node features to use, stored as a pickled numpy
                        array.
  -t TARGET, --target TARGET
                        The target edge attribute, default is 'score'
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
from typing import AnyStr, Any, List, Tuple, Dict, Optional

from keras import backend as K
from keras import Input, Model, optimizers, losses, activations
from keras.layers import Layer, Dense, Concatenate, Multiply, Activation, Lambda
from keras.utils import Sequence

from stellar.layer.hinsage import Hinsage


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
        self.idx = 0
        self.name = name
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if (self.idx >= self.data_size):
            print("this shouldn't happen, but it does")
            self.on_epoch_end()
        print("Fetching {} batch {}".format(self.name, str(self.idx)))

        # Get edges and labels
        idx_end = self.idx + self.batch_size
        edges = self.ids[self.idx:idx_end]
        batch_labels = self.labels[self.idx:idx_end]

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

        # Advance data index
        self.idx = idx_end

        return batch_feats, batch_labels

    def on_epoch_end(self):
        'Resets index and shuffles data'
        self.idx = 0
        shuffle_inx = random.sample(range(self.data_size), self.data_size)
        self.ids = [self.ids[i] for i in shuffle_inx]
        self.labels = np.array([self.labels[i] for i in shuffle_inx])


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

        # Default config settings
        self.input_feature_size: int = self.features.shape[1]
        self.batch_size: int = 1000

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
            ('user', [2]), ('movie', [3]),
            ('movie', [4]), ('user', [5]),
            ('user', []), ('movie', []),
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
        self.labels_train = np.array([
            G.edges[e][target_name] for e in train_edges])
        self.labels_test = np.array([
            G.edges[e][target_name] for e in test_edges])
        print("done!")

    def shapes_from_schema(self) -> List[Tuple[int,int]]:
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
            return (
                np.product(self.node_samples[:i], dtype=int),
                self.input_feature_size
            )

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

    def sample_neighbours(self, indices: List[int], ns: int,
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
            adj[inx][i]
            for inx in indices
            for i in np.random.randint(len(adj[inx]), size=ns)
        ]

    def edge_data_generators(self):
        train_gen = EdgeDataGenerator(self, self.edgelist_train,
                                      self.labels_train, self.node_samples,
                                      self.batch_size, 'train')

        test_gen = EdgeDataGenerator(self, self.edgelist_test,
                                     self.labels_test, self.node_samples,
                                     self.batch_size, 'test')
        return train_gen, test_gen


class RecommenderPrediction(Layer):
    def __init__(self, method='ip', use_bias=False, act='relu', **kwargs):
        self.method = method
        self.use_bias = use_bias
        self.activation = activations.get(act)
        self.output_dim = 1
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.method!='ip':
            self.W = self.add_weight(
                name='regression',
                shape=self.compute_output_shape(input_shape),
                initializer='glorot',
                trainable=True)
        if self.use_bias:
            pass
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.method == 'concat':
            xint = K.concatenate(x, axis=-1)
        else:
            xint = K.prod(x, axis=-1)

        if self.method == 'ip':
            out = K.sum(xint, axis=-1)
        else:
            out = K.dot(xint, self.W)
            if self.use_bias:
                out = K.bias_add(out, self.bias)
            if self.activation is not None:
                out = self.activation(out)

        # Reduce dims
        return K.squeeze(out, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def regression_pred_mul(x):
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph.

    The edge feature is formed from a Hadamard product of transformed
    movie and user embeddings.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(32, activation='relu')(x[0])
    x1 = Dense(32, activation='relu')(x[1])
    le = Multiply()([x0, x1])

    return Dense(1, activation='linear')(le)

def regression_pred_concat(x):
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph

    The edge feature is formed from a concatenation of transformed movie
    and user embeddings followed by a dense NN layer.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(16, activation='relu')(x[0])
    x1 = Dense(16, activation='relu')(x[1])
    le = Concatenate()([x0, x1])
    le = Dense(32, activation='relu')(le)

    return Dense(1, activation='linear')(le)


def regression_pred_ip(x):
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph

    This is a direct inner product between the user and movie embedddings.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(16, activation='relu')(x[0])
    x1 = Dense(16, activation='relu')(x[1])
    return Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))(
        [x0, x1])


def train(ml_data: MovielensData, batch_size: int = 1000, num_epochs: int = 10):
    """
    Main training loop
    """
    # Select the edge regressor to use
    if ml_data.edge_regressor == "ip":
        regression_pred = regression_pred_ip

    elif ml_data.edge_regressor == "concat":
        regression_pred = regression_pred_concat

    elif ml_data.edge_regressor == "mul":
        regression_pred = regression_pred_mul

    # Create data iterator
    train_iter, test_iter = ml_data.edge_data_generators()

    # HINSAGE model
    hs = Hinsage(
        output_dims=ml_data.layer_sizes,
        n_samples=ml_data.node_samples,
        input_neigh_tree=ml_data.subtree_schema,
        input_dim={'user': ml_data.input_feature_size,
                   'movie': ml_data.input_feature_size,},
        bias=ml_data.use_bias,
        dropout=ml_data.dropout)

    x_inp = [Input(shape=s) for s in ml_data.shapes_from_schema()]
    x_out = hs(x_inp)

    print("outputs", x_out)

    # Final estimator layer
    #rl = RecommenderPrediction(method='ip', act='relu')
    pred = regression_pred(x_out)

    print("final model:", pred)

    model = Model(inputs=x_inp, outputs=pred)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.mean_squared_error,
        metrics=['accuracy'])


    model.fit_generator(train_iter, epochs=1, verbose=2)



def test(ml_data: MovielensData, model_file: AnyStr):
    """
    Predict and measure the test performance
    """
    pass


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description="Run GraphSAGE on movielens")
    parser.add_argument(
        '-c', '--checkpoint', nargs='?', type=str, default=None,
        help="Load a save checkpoint file")
    parser.add_argument(
        '-n', '--batch_size', type=int, default=500,
        help="Load a save checkpoint file")
    parser.add_argument(
        '-e', '--epochs', type=int, default=10,
        help="Number of epochs to train for")
    parser.add_argument(
        '-s', '--neighbour_samples', type=int, nargs='*', default=[30, 10],
        help="The number of nodes sampled at each layer")
    parser.add_argument(
        '-l', '--layer_size', type=int, nargs='*', default=[50, 50],
        help="The number of hidden features at each layer")
    parser.add_argument(
        '-m', '--method', type=str, default='ip',
        help="The edge regression method: 'concat', 'mul', or 'ip")
    parser.add_argument(
        '-b', '--baseline', action='store_true',
        help="Use a learned offset for each node.")
    parser.add_argument(
        '-g', '--graph', type=str,
        default='data/ml-1m_split_graphnx.pkl',
        help="The graph stored in networkx pickle format.")
    parser.add_argument(
        '-f', '--features', type=str, default='data/ml-1m_embeddings.pkl',
        help="The node features to use, stored as a pickled numpy array.")
    parser.add_argument(
        '-t', '--target', type=str, default='score',
        help="The target edge attribute, default is 'score'")
    args, cmdline_args = parser.parse_known_args()
    # yapf: enable

    print("Running GraphSAGE recommender:")

    graph_loc = os.path.expanduser(args.graph)
    features_loc = os.path.expanduser(args.features)

    ml_data = MovielensData(graph_loc, features_loc, args.target)

    # Training: batch size & epochs
    batch_size = args.batch_size
    num_epochs = args.epochs

    # number of samples per additional layer,
    ml_data.node_samples = args.neighbour_samples

    # number of features per additional layer
    ml_data.layer_size = args.layer_size

    # The edge regressor to use
    ml_data.edge_regressor = args.method

    # Per-node baselines - learns a baseline for movies and users
    # requires fixed set of train/test movies + users
    ml_data.node_baseline = args.baseline

    if args.checkpoint is None:
        train(ml_data, batch_size, num_epochs)
    else:
        test(ml_data, args.checkpoint)
