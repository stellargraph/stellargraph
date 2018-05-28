"""
Helper functions to write MovieLens data to Redis prior to training.

"""
import sys
import os
import pickle
import numpy as np
from redis import StrictRedis
import networkx as nx
import collections
import pandas as pd
import tensorflow as tf

from model.hinsage import hinsage_supervised, MeanAggregatorHin, Schema
from graph.redisgraph import RedisHin


class MovielensRedis:
    """
    Class to manage movielens graph data and write to redis.

    Args:
        graph_loc: Location of networkx graph file
        features_loc: Location of features file
        target_name: Name of edge target attribute to predict.
    """

    def __init__(self, graph_loc, features_loc, target_name):
        # Read graph
        print("Reading graph...")
        G = nx.read_gpickle(graph_loc)
        self.n_nodes = G.number_of_nodes()

        # Read features
        print("Reading features...")
        feats = self.read_feats(features_loc)

        assert feats.shape[0] == self.n_nodes

        # Default config settings
        self.input_feature_size = feats.shape[1]
        self.batch_size = 1000

        # number of samples per additional layer,
        self.node_samples = [30, 10]

        # number of features per additional layer
        self.layer_sizes = [50, 50]

        # Per-node baselines - learns a baseline for movies and users
        # requires fixed set of train/test movies + users
        self.node_baseline = True

        # Layer bias
        self.use_bias = True

        # Directed edgelist
        # Note that this assumes that the movie IDs are lower than the
        # user IDs
        self.edgelist_train = [
            (min(e), max(e)) for e in G.edges if G.edges[e]['split'] == 0
        ]
        self.edgelist_test = [
            (min(e), max(e)) for e in G.edges if G.edges[e]['split'] == 1
        ]

        # Create list of target labels
        self.labels_train = [
            G.edges[e][target_name] for e in G.edges if G.edges[e]['split'] == 0
        ]
        self.labels_test = [
            G.edges[e][target_name] for e in G.edges if G.edges[e]['split'] == 1
        ]

        # Directed adjacency lists
        # Note that test adjacency lists contain both train & test edges
        adjlist_train_u, adjlist_train_m = self.nx_to_adjlists(G, split=0)
        adjlist_test_u, adjlist_test_m = self.nx_to_adjlists(G, split=None)

        # Store the adjacency lists in Python for faster sampling
        self.adj_lists = {
            "train::USM": adjlist_train_u,
            "train::MSU": adjlist_train_m,
            "test::USM": adjlist_test_u,
            "test::MSU": adjlist_test_m,
        }

        print("Writing to Redis...", end="")
        sys.stdout.flush()

        # redis instance at localhost:6379
        self._r = StrictRedis(port=6379)

        # write features
        self.write_features(feats)

        # Write adjacency lists for each edge type
        # Each adjacency list encoded in redis with train/test,
        #  ID and edge type e.g. { "train:1234:USM" : [1111, 2222] }
        # Edge types defined as:
        #   MSU: Movie - ScoredBy -> User
        #   USM: User - Scored -> Movie
        # write_adj(adjlist_train_u, "test:", ":USM")
        # write_adj(adjlist_train_m, "test:", ":MSU")

        # write_adj(adjlist_test_u, "train:", ":USM")
        # write_adj(adjlist_test_m, "train:", ":MSU")

        # Write test IDs to list
        self.write_id_list("test", self.edgelist_test)

        # Write labels and traversable IDs (USM edges only)
        self.write_labels(self.labels_train, self.edgelist_train)

        # Shuffle training set and write train IDs to list
        self.write_id_shuffle("train", self.edgelist_train)

        print("done!")

    def info_str(self):
        """
        String with model parameters for saving files.
        """
        attr_str = "N" if self.node_baseline else ""
        attr_str += "B" if self.use_bias else ""

        return "ns{}-{}_nf{}-{}-{}_{}".format(
            self.node_samples[0], self.node_samples[1], self.input_feature_size,
            self.layer_sizes[0], self.layer_sizes[1], attr_str
        )

    def write_id_shuffle(self, name, edgelist):
        """
        Write a list of values in random order to redis.
        If edgelist is None it will shuffle the existing values.

        Args:
            name: Name of list
            edgelist: List of edges to be written.

        """
        assert isinstance(edgelist, collections.Iterable)

        if edgelist:
            elcp = edgelist.copy()
        else:
            elcp = [pickle.loads(i) for i in self._r.lrange(name, 0, -1)]

        np.random.shuffle(elcp)
        self._r.delete(name)
        self._r.lpush(name, *[pickle.dumps(el) for el in elcp])

    def write_id_list(self, name, edgelist):
        """
        Write a list of values to redis. The list will be written
        in the same order as supplied.

        Args:
            name: Name of list
            edgelist: List of edges to be written.
        """
        assert isinstance(edgelist, collections.Iterable)

        self._r.delete(name)
        self._r.lpush(name, *[pickle.dumps(el) for el in edgelist[::-1]])

    def write_id_set(self, name, items):
        """
        Write a set of values to redis.

        Args:
            name: Name of set
            items: List of items to be written.
        """
        self._r.delete(name)
        self._r.sadd(name, *[pickle.dumps(el) for el in items])

    def nx_to_adjlists(self, G, split=0):
        adj_forward = {}
        adj_reverse = {}
        for n, nbrdict in G.adjacency():
            neighs = [
                k for k, edge_info in nbrdict.items()
                if split is None or (edge_info['split'] == split)
            ]
            # If no edges in selected split, continue
            if len(neighs) == 0:
                continue
            # Add u->m links to forward lists
            if n < neighs[0]:
                adj_forward[n] = neighs
            # Add m->u links to reverse lists
            else:
                adj_reverse[n] = neighs
        return adj_forward, adj_reverse

    def write_features(self, features):
        pipe = self._r.pipeline()
        for i in range(features.shape[0]):
            pipe.set("feat:" + str(i), features[i].astype(np.float32).ravel().tostring())
        pipe.execute()

    def write_adj(self, adj_lists, prefix, suffix):
        pipe = self._r.pipeline()
        for src, dst in adj_lists.items():
            pipe.delete(prefix + str(src) + suffix)
            pipe.sadd(prefix + str(src) + suffix, *dst)
        pipe.execute()

    def write_labels(self, labels, edgelist, onehot=False):
        pipe = self._r.pipeline()
        if onehot:
            n_labels = max(labels)  # assuming no zeros
            labels_onehot = np.eye(n_labels)[[lb-1 for lb in labels]]
            for lb, edge in zip(labels_onehot, edgelist):
                pipe.set('label:' + str(edge), pickle.dumps(lb))
            pipe.set('num_labels', n_labels)
        else:
            for lb, edge in zip(labels, edgelist):
                pipe.set('label:' + str(edge), pickle.dumps([lb]))
            pipe.set('num_labels', 1)
        pipe.execute()

    def read_edgelist(self, fn):
        lines = np.loadtxt(fn, delimiter=" ", unpack=False)
        return [(line[0], line[1]) for line in lines.astype(np.int32)]

    def read_labels(self, fn):
        with open(fn, 'r') as f:
            return [int(lb) for lb in f.read().split()]

    def read_feats(self, fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)

    def calc_test_metrics(self, predictions):
        pred_test = np.ravel(predictions)
        labels_test = np.ravel(self.labels_test)

        assert pred_test.shape == labels_test.shape

        # Calculate MAE error
        rmse = np.sqrt(np.mean((pred_test - labels_test)**2))
        mae = np.mean(np.abs(pred_test - labels_test))

        print("Test error:")
        print("RMSE: {}".format(rmse))
        print("MAE: {}".format(mae))

        # Calculate precision/recall
        true_relevant = labels_test > 3
        true_irrelevant = labels_test <= 3
        pred_relevant = pred_test > 3
        pred_irrelevant = pred_test <= 3

        tp = np.sum(true_relevant * pred_relevant)
        fp = np.sum(true_irrelevant * pred_relevant)
        fn = np.sum(true_relevant * pred_irrelevant)
        tn = np.sum(true_irrelevant * pred_irrelevant)

        print("Confusion matrix:")
        print([[tp,fp],[fn,tn]])

        print("Precision:", tp/(tp + fp))
        print("Recall:", tp/(tp + fn))

    def save_predictions(self, predictions, filename):
        pred_test = np.ravel(predictions)
        labels_test = np.ravel(self.labels_test)

        pd.DataFrame({
            "pred_test": pred_test,
            "true_test": labels_test
        }).to_csv(
            filename,
            index=False
        )

    def create_schema(self):
        # number of samples per additional layer,
        ns = self.node_samples

        # number of features per additional layer
        nf = [self.input_feature_size] + self.layer_sizes

        # create schema for HinSAGE
        self.schema = Schema(
            types=['user', 'movie', 'movie', 'user', 'user', 'movie'],
            n_samples=[n for n in ns+[1] for _ in range(2)][::-1],
            neighs=[[(2, 'USM')], [(3, 'MSU')], [(4, 'MSU')], [(5, 'USM')], [], []],
            dims=[{'user': (d, 1), 'movie': (d, 1)} for d in nf],
            n_layers=2,
            xlen=[6, 4, 2]
        )
        return self.schema

    def create_iterators(self, batch_size=1000):
        """
        Creates a Tensorflow iterator and its initializers with given graph object.
        Tensorflow initializer objects are used to initialize the iterator with
        either the training or the test set.

        :param graph:   Graph object containing generator methods for traversing through
                        train and test sets
        :param schema:  Graph and sampling schema
        :return: Tuple of batch iterator, training set initializer, test set initializer
        """
        # data graph
        graph = RedisHin(StrictRedis(), adj_lists=self.adj_lists)

        # input types
        inp_types = (tf.int32, tf.int32, tf.int32, tf.float32,
                     *[tf.float32]*self.schema.xlen[0])

        # input shapes
        inp_shapes = (
            tf.TensorShape(()),
            tf.TensorShape((None)),
            tf.TensorShape((None)),
            tf.TensorShape((None, 1)),
            *[tf.TensorShape((None, self.schema.dims[0][t][0])) for t in self.schema.types]
        )

        # train and test data
        ds_train = tf.data.Dataset.from_generator(
            graph.train_gen(batch_size, self.schema), inp_types, inp_shapes
        ).prefetch(1)
        ds_test = tf.data.Dataset.from_generator(
            graph.test_gen(batch_size, self.schema), inp_types, inp_shapes
        )

        tf_batch_iter = tf.data.Iterator.from_structure(inp_types, inp_shapes)
        return (
            tf_batch_iter,
            tf_batch_iter.make_initializer(ds_train),
            tf_batch_iter.make_initializer(ds_test)
        )
