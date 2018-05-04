"""
Helper functions to write MovieLens data to Redis prior to training.

"""

import pickle
import numpy as np
import os
from redis import StrictRedis
from functools import reduce


def write_id_shuffle(r, name, edgelist):
    elcp = edgelist.copy() if edgelist is not None else [pickle.loads(i) for i in r.lrange(name, 0, -1)]
    np.random.shuffle(elcp)
    r.delete(name)
    r.lpush(name, *[pickle.dumps(el) for el in elcp])


def write_to_redis(path):
    def write_features(r, features):
        pipe = r.pipeline()
        for i in range(features.shape[0]):
            pipe.set("feat:" + str(i), features[i].astype(np.float32).ravel().tostring())
        pipe.execute()

    def write_adj(r, edgelist, prefix, suffix):
        pipe = r.pipeline()
        adj_lists = reduce(
            lambda x, y: {**x, **{y[0]: x[y[0]] + [y[1]] if y[0] in x else [y[1]]}},
            edgelist,
            {}
        )
        for src, dst in adj_lists.items():
            pipe.delete(prefix + str(src) + suffix)
            pipe.sadd(prefix + str(src) + suffix, *dst)
        pipe.execute()

    def write_labels(r, labels, edgelist, onehot=False):
        pipe = r.pipeline()
        if onehot:
            n_labels = max(labels)  # assuming no zeros
            labels_onehot = np.eye(n_labels)[[lb-1 for lb in labels]]
            for lb, edge in zip(labels_onehot, edgelist):
                pipe.set('label:' + str(edge), pickle.dumps(lb))
            pipe.set('num_labels', n_labels)
        else:
            [pipe.set('label:' + str(edge), pickle.dumps([lb])) for lb, edge in zip(labels, edgelist)]
            pipe.set('num_labels', 1)
        pipe.execute()

    def write_id_set(r, name, edgelist):
        r.delete(name)
        r.sadd(name, *[pickle.dumps(el) for el in edgelist])

    def read_edgelist(fn):
        lines = np.loadtxt(fn, delimiter=" ", unpack=False)
        return [(line[0], line[1]) for line in lines.astype(np.int32)]

    def read_labels(fn):
        with open(fn, 'r') as f:
            return [int(lb) for lb in f.read().split()]

    def read_feats(fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)

    # read edge list
    edgelist = read_edgelist(os.path.join(path, "ml_100k_edge_homogeneous.txt"))
    edgelist_test, edgelist_train = edgelist[:20000], edgelist[20000:]

    # read labels
    labels = read_labels(os.path.join(path, 'ml_100k_edge_labels.txt'))

    # read features
    feats = read_feats(os.path.join(path, 'embeddings.pkl'))

    # redis instance at localhost:6379
    r = StrictRedis()

    # write features
    write_features(r, feats)

    # Write adjacency lists for each edge type
    # Each adjacency list encoded in redis with train/test, ID and edge type e.g. { "train:1234:USM" : [1111, 2222] }
    # Edge types defined as:
    #   MSU: Movie - ScoredBy -> User
    #   USM: User - Scored -> Movie
    write_adj(r, edgelist, "test:", ":USM")
    write_adj(r, [el[::-1] for el in edgelist], "test:", ":MSU")
    write_adj(r, edgelist_train, "train:", ":USM")
    write_adj(r, [el[::-1] for el in edgelist_train], "train:", ":MSU")

    # write labels and traversable IDs (USM edges only)
    write_labels(r, labels, edgelist)
    write_id_shuffle(r, "train", edgelist_train)
    write_id_set(r, "test", edgelist_test)
