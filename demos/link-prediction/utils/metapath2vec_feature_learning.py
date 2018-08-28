# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
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
from gensim.models import Word2Vec
import time
import pandas as pd
from stellar.data.explorer import UniformRandomMetaPathWalk


class Metapath2VecFeatureLearning(object):
    def __init__(self, nxG=None, embeddings_filename=r"metapath2vec_model.emb"):
        self.nxG = nxG
        self.G = None
        self.model = None
        self.embeddings_filename = embeddings_filename

    #
    # learn_embeddings() and fit() are based on the learn_embeddings() and main() functions in main.py of the
    # reference implementation of Node2Vec.
    #
    def learn_embeddings(self, walks, d, k):
        """
        Learn embeddings by optimizing the Skipgram objective using SGD.

        Args:
            walks:
            d:
            k:

        Returns:

        """
        time_b = time.time()
        walks = [list(map(str, walk)) for walk in walks]
        self.model = Word2Vec(
            walks, size=d, window=k, min_count=0, sg=1, workers=2, iter=1
        )
        self.model.wv.save_word2vec_format(self.embeddings_filename)
        print(
            "({}) Time to learn embeddings {:.0f} seconds".format(
                type(self).__name__, time.time() - time_b
            )
        )

    def _assert_positive_int(self, val, msg=""):
        """
        Raises ValueError exception if val is not a positive integer.

        Args:
            val: The value to check
            msg: The message to return with the exception

        Returns:

        """
        if val <= 0 or not isinstance(val, int):
            raise ValueError(msg)

    def _assert_positive(self, val, msg=""):
        """
        Raises ValueError exception if val is not a positive number.

        Args:
            val: The value to check
            msg: The message to return with the exception

        Returns:

        """
        if val <= 0:
            raise ValueError(msg)

    def fit(self, metapaths=None, d=128, r=10, l=80, k=10):
        """
        Pipeline for representational learning for all nodes in a graph.

        :param k:
        :return:
        """
        self._assert_positive_int(d, msg="d should be positive integer")
        self._assert_positive_int(r, msg="r should be positive integer")
        self._assert_positive_int(l, msg="l should be positive integer")
        self._assert_positive_int(k, msg="k should be positive integer")

        start_time_fit = time.time()
        # self.G = node2vec.Graph(self.nxG, False, p, q)
        # self.G.preprocess_transition_probs()
        metapath_walker = UniformRandomMetaPathWalk(self.nxG)
        # walks = self.G.simulate_walks(r, l)
        time_b = time.time()
        walks = metapath_walker.run(
            nodes=list(self.nxG.nodes()),
            metapaths=metapaths,
            length=l,
            n=r,
            node_type_attribute="label",
            seed=None,
        )
        print(
            "({}) Time for random walks {:.0f} seconds.".format(
                type(self).__name__, time.time() - time_b
            )
        )
        self.learn_embeddings(walks, d, k)
        print("Total time for fit() was {:.0f}".format(time.time() - start_time_fit))

    def from_file(self, filename):
        """
        Helper function for loading a node2vec model from disk so that I can run experiments fast without having to
        wait for node2vec to finish.

        :param filename: The filename storing the model
        :return:  None
        """
        self.model = pd.read_csv(filename, delimiter=" ", skiprows=1, header=None)
        self.model.iloc[:, 0] = self.model.iloc[:, 0].astype(
            str
        )  # this is so that indexing works the same as having
        # trained the model with self.fit()
        self.model.index = self.model.iloc[:, 0]
        self.model = self.model.drop([0], 1)
        print(self.model.head(2))

    def select_operator_from_str(self, binary_operator):
        if binary_operator == "l1":
            return self.operator_l1
        elif binary_operator == "l2":
            return self.operator_l2
        elif binary_operator == "avg":
            return self.operator_avg
        elif binary_operator == "h":  # hadamard
            return self.operator_hadamard
        else:
            raise ValueError("Invalid binary operator {}".format(binary_operator))

    def operator_hadamard(self, u, v):
        return u * v

    def operator_avg(self, u, v):
        return (u + v) / 2.0

    def operator_l2(self, u, v):
        return (u - v) ** 2

    def operator_l1(self, u, v):
        return np.abs(u - v)

    def transform(self, edge_data, binary_operator="h"):
        """
        It calculates edge features for the given binary operator applied to the node features in data_edge

        :param edge_data: (2-tuple) It is a list of pairs of nodes that make an edge in the graph
        :param binary_operator: The binary operator to apply to the node features to calculate an edge feature
        :return: Features in X (Nxd array where N is the number of edges and d is the dimensionality of the edge
            features that is the same as the dimensionality of the node features) and edge labels in y (0 for no edge
            and 1 for edge).
        """
        X = []  # data matrix, each row is a d-dimensional feature of an edge

        func_bin_operator = self.select_operator_from_str(binary_operator)

        for ids in edge_data[0]:
            u_str = str(ids[0])
            v_str = str(ids[1])
            if type(self.model) is Word2Vec:
                X.append(func_bin_operator(self.model[u_str], self.model[v_str]))
            else:  # Pandas Dataframe
                X.append(
                    func_bin_operator(self.model.loc[u_str], self.model.loc[v_str])
                )

        return np.array(X), edge_data[1]
