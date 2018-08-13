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
Graph link attribute prediction using HinSAGE, using the movielens data.
"""

import argparse
import pickle
from keras import Input, Model, optimizers, losses, activations, metrics
from stellar.data.stellargraph import *
from stellar.mapper.link_mappers import *
from stellar.layer.hinsage import *
from stellar.layer.link_inference import link_regression
from typing import AnyStr, List, Dict


def read_graph(graph_fname, features_fname):

    # Read graph
    print("Reading graph...")
    gnx = nx.read_gpickle(graph_fname)

    # Read features
    print("Reading features...")
    with open(features_fname, "rb") as f:
        features = pickle.load(f)

    #  Convert to StellarGraph:
    if gnx.is_directed():
        g = StellarDiGraph(gnx)
    else:
        g = StellarGraph(gnx)

    # Add features to nodes in g:
    nx.set_node_attributes(g, values=dict(zip(list(g), features)), name="feature")
    g.node_feature_size = features.shape[1]

    print(
        "Graph statistics: {} nodes, {} edges".format(
            g.number_of_nodes(), g.number_of_edges()
        )
    )

    return g


def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


class LinkInference(object):
    """
    Link attribute inference class
    """

    def __init__(self, g: StellarGraphBase):
        self.g = g

    def train(
        self,
        layer_size: List[int],
        num_samples: List[int],
        batch_size: int = 1000,
        num_epochs: int = 10,
        learning_rate=0.005,
        use_bias=True,
        dropout=0.0,
    ):
        """
        Build and train the HinSAGE model for link attribute prediction on the specified graph G
        with given parameters.

        Args:
            layer_size:
            num_samples:
            batch_size:
            num_epochs:
            learning_rate:
            dropout:

        Returns:

        """

        # Training and test edges
        edges_train = [e for e in self.g.edges(data=True) if e[2]["split"] == 0]
        edges_test = [e for e in self.g.edges(data=True) if e[2]["split"] == 1]

        #  Edgelists:
        edgelist_train = [(e[0], e[1]) for e in edges_train]
        edgelist_test = [(e[0], e[1]) for e in edges_test]

        # Directed ('movie', 'user') edgelists:
        # Note that this assumes that the movie IDs are lower than the user IDs
        edgelist_train = [(min(e), max(e)) for e in edgelist_train]
        edgelist_test = [(min(e), max(e)) for e in edgelist_test]

        # !HACK: node types should normally be in g already! Add node types to self.g:
        movie_nodes = np.unique([e[0] for e in edgelist_train + edgelist_test])
        user_nodes = np.unique([e[1] for e in edgelist_train + edgelist_test])
        node_types = {}
        [node_types.update({v: "movie"}) for v in movie_nodes]
        [node_types.update({v: "user"}) for v in user_nodes]
        nx.set_node_attributes(self.g, name="label", values=node_types)

        labels_train = [e[2]["score"] for e in edges_train]
        labels_test = [e[2]["score"] for e in edges_test]

        # Link mappers:
        mapper_train = HinSAGELinkMapper(
            self.g,
            edgelist_train,
            labels_train,
            batch_size,
            num_samples,
            name="mapper_train",
        )
        mapper_test = HinSAGELinkMapper(
            self.g,
            edgelist_test,
            labels_test,
            batch_size,
            num_samples,
            name="mapper_test",
        )

        assert mapper_train.type_adjacency_list == mapper_test.type_adjacency_list

        # Model:
        hinsage = Hinsage(
            output_dims=layer_size,
            n_samples=num_samples,
            input_neigh_tree=mapper_train.type_adjacency_list,
            input_dim={
                "user": self.g.node_feature_size,
                "movie": self.g.node_feature_size,
            },
            bias=use_bias,
            dropout=dropout,
        )

        # Define input and output sockets of hinsage:
        mapper_tmp = HinSAGELinkMapper(
            self.g,
            [edgelist_train[0]],
            [labels_train[0]],
            1,
            num_samples,
            name="mapper_tmp",
        )

        feats, l = mapper_tmp.__getitem__(0)  # get features, labels from the mapper
        input_shapes = []
        for f in feats:
            input_shapes.append(f.shape[1:])
        x_inp = [Input(shape=s) for s in input_shapes]
        x_out = hinsage(x_inp)

        # Final estimator layer
        score_prediction = link_regression(
            edge_feature_method=args.edge_feature_method
        )(x_out)

        # Create Keras model for training
        model = Model(inputs=x_inp, outputs=score_prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=learning_rate),
            loss=losses.mean_squared_error,
            metrics=[root_mean_square_error, metrics.mae],
        )

        # Train model
        history = model.fit_generator(
            mapper_train,
            validation_data=mapper_test,
            epochs=num_epochs,
            verbose=2,
            shuffle=True,
        )

        # Evaluate and print metrics
        test_metrics = model.evaluate_generator(mapper_test)

        print("Test Evaluation:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

    def test(self, G: StellarGraphBase, model_file: AnyStr):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GraphSAGE on movielens")

    parser.add_argument(
        "-g",
        "--graph",
        type=str,
        default="data/ml-1m_split_graphnx.pkl",
        help="The graph stored in networkx pickle format.",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="data/ml-1m_embeddings.pkl",
        help="The node features to use, stored as a pickled numpy array.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="score",
        help="The target edge attribute, default is 'score'",
    )
    parser.add_argument(
        "-m",
        "--edge_feature_method",
        type=str,
        default="ip",
        help="The method for combining node embeddings into edge embeddings: 'concat', 'mul', or 'ip",
    )
    parser.add_argument(
        "-r",
        "--learningrate",
        type=float,
        default=0.0005,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "-n", "--batch_size", type=int, default=500, help="Load a save checkpoint file"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-s",
        "--neighbour_samples",
        type=int,
        nargs="*",
        default=[30, 10],
        help="The number of nodes sampled at each layer",
    )
    parser.add_argument(
        "-l",
        "--layer_size",
        type=int,
        nargs="*",
        default=[50, 50],
        help="The number of hidden features at each layer",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout for the HinSAGE model, between 0.0 and 1.0",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        nargs="?",
        type=str,
        default=None,
        help="Load a checkpoint file",
    )

    args, cmdline_args = parser.parse_known_args()

    G = read_graph(args.graph, args.features)

    model = LinkInference(G)

    if args.checkpoint is None:
        model.train(
            learning_rate=args.learningrate,
            layer_size=args.layer_size,
            num_samples=args.neighbour_samples,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            dropout=args.dropout,
        )
    else:
        model.test(args.checkpoint)
