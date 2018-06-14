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
from keras import Input, Model, optimizers, losses, activations, metrics
from keras.layers import Layer, Dense, Concatenate, Multiply, Activation, Lambda
from keras.utils import Sequence

from recommender import (MovielensData, root_mean_square_error,
                         regression_predictor)


def train(ml_data: MovielensData, batch_size: int = 1000,
          num_epochs: int = 10):
    # Create data iterators
    train_iter, test_iter = ml_data.edge_data_generators(batch_size)

    # HINSAGE model
    hs = ml_data.create_model()

    x_inp = [Input(shape=s) for s in ml_data.shapes_from_schema()]
    x_out = hs(x_inp)

    # Final estimator layer
    score_prediction = regression_predictor(
        hidden_1=32, hidden_2=32,
        method=ml_data.edge_regressor,
        )(x_out)

    # Create Keras model for training
    model = Model(inputs=x_inp, outputs=score_prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=0.0005),
        loss=losses.mean_squared_error,
        metrics=[root_mean_square_error, metrics.mae])

    # Train model
    history = model.fit_generator(
        train_iter, epochs=num_epochs, verbose=2, shuffle=True)

    # Evaluate and print metrics
    test_metrics = model.evaluate_generator(test_iter)

    print("Test Evaluation:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


def test(ml_data: MovielensData, model_file: AnyStr):
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
