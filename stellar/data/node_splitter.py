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

import os
import uuid
import numpy as np
import pandas as pd

from stellar.data.stellargraph import StellarGraphBase
from stellar.data.epgm import EPGM

UNKNOWN_TARGET_ATTRIBUTE = "-1"

# Easier functional interface for the splitter:
def train_val_test_split(
    G: StellarGraphBase,
    node_type=None,
    test_size=0.4,
    train_size=0.2,
    stratify=False,
    seed=None,
):
    """
        Splits node data into train, test, validation, and unlabeled sets.

        Any nodes that have a target value equal to UNKNOWN_TARGET_ATTRIBUTE are added to the unlabeled set.

        The validation set includes all nodes that remain after the train, test and unlabeled sets have been
        created. As a result, it is possible the the validation set is empty.

    Args:
        G : StellarGraph containing the nodes to be split.

        test_size: float, int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the test split. If int, represents the absolute number of test samples.

        train_size: float, int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the train split. If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

        seed: int or None, optional (default=None)
            If this is an int the seed will be used to initialize a random number generator,
            otherwith the numpy default will be used.

        shuffle : boolean, optional (default=True)
            Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

        stratify: bool
            If True the data is split in a stratified fashion with equal numbers of nodes taken
            for each target category.

    Returns:
            y_train, y_val, y_test, y_unlabeled
    """
    node_splitter = NodeSplitter()

    # We will need to get the darn labels regardless of if we have stratification
    nodes = G.get_nodes_of_type(node_type)
    labels = G.get_raw_targets(
        nodes, node_type, unlabeled_value=UNKNOWN_TARGET_ATTRIBUTE
    )

    # Number of nodes and number without a label
    n_nodes = len(nodes)
    n_known = sum(l != UNKNOWN_TARGET_ATTRIBUTE for l in labels)

    if n_known == 0:
        raise RuntimeError("No nodes with target attribute to split.")

    # Find the number of nodes to use in the training set
    if isinstance(train_size, float) and (0 < train_size <= 1):
        train_size_n = int(n_known * train_size)

    elif isinstance(train_size, int):
        train_size_n = train_size

    else:
        raise ValueError("Splitter: train_size should be specified as a float or int")

    # Find the number of nodes to use in the test set
    if isinstance(test_size, float) and (0 < test_size <= 1):
        test_size_n = int(n_known * test_size)

    elif isinstance(test_size, int):
        test_size_n = test_size

    else:
        raise ValueError("Splitter: train_size should be specified as a float or int")

    # Find the number of nodes to use in the validation set
    val_size = None
    if isinstance(val_size, float) and (0 < val_size <= 1):
        val_size_n = int(n_known * val_size)

    elif isinstance(val_size, int):
        val_size_n = val_size

    else:
        val_size_n = max(0, n_known - (train_size_n + test_size_n))

    # Check that these sizes make sense
    if (train_size_n + test_size_n + val_size_n) > n_known:
        raise ValueError(
            "Number of train, test and val nodes "
            "is greater than the total number of labelled nodes."
        )

    # Now the splitter needs the node IDs and labels zipped together
    # TODO: This is a hack as the splitter only works when this array is sting type
    nodeid_and_label = np.array([nl for nl in enumerate(labels)], dtype="U")

    # If stratified sampling, we need the target labels.
    if stratify:
        class_set = set(labels)

        # Remove the unknown target type
        class_set.discard(UNKNOWN_TARGET_ATTRIBUTE)

        # The number of classes we have
        n_classes = len(class_set)

        # The number of nodes we want per class
        p = int(train_size_n / n_classes)

        splits = node_splitter.train_test_split(
            y=nodeid_and_label, p=p, method="count", test_size=test_size_n, seed=seed
        )

    else:
        splits = node_splitter.train_test_split(
            y=nodeid_and_label,
            method="absolute",
            train_size=train_size_n,
            test_size=test_size_n,
            seed=seed,
        )

    # Get the node_ids out of the splitter
    node_ids_out = [[nodes[int(ind)] for ind in split[:, 0]] for split in splits]

    return node_ids_out


class NodeSplitter(object):
    def __init__(self):
        self.format_epgm = False
        self.g_epgm = None
        self.g_id = None

    def _get_nodes(self, graph_nodes, node_type, target_attribute):
        """
        Returns a list of node IDs for the subset of graph_nodes that have the given node type.
        :param graph_nodes: <list> List of OrderedDict with vertex data for graph in EPGM format
        :param node_type: <str> The node type of interest
        :param target_attribute: <str> The target attribute key
        :return: <list> List of node IDs that have given node type
        """
        # This code will fail if a node of node_type is missing the target_attribute.
        # We can fix this by using node['data'].get(target_attribute, None) so that at least all nodes of the
        # given type are returned. However, we must check for None in target_attribute later to exclude these nodes
        # from being added to train, test, and validation datasets.
        # y = [(node['id'], node['data'][target_attribute]) for node in graph_nodes if node['meta']['label'] == node_type]
        # TODO: Change this to handler user speficied node type attribute name, not just assume "label"
        y = [
            (node["id"], node["data"].get(target_attribute, UNKNOWN_TARGET_ATTRIBUTE))
            for node in graph_nodes
            if node["meta"]["label"] == node_type
        ]

        return y

    def load_data(self, path, dataset_name=None, node_type=None, target_attribute=None):
        """
        Loads the node data

        :param path: Input filename or directory where graph in EPGM format is stored
        :param node_type: For HINs, the node type to consider
        :param target_attribute: For EPGM format, the target node attribute
        :return: N x 2 numpy arrays where the first column is the node id and the second column is the node label.
        """
        if os.path.isdir(path):
            self.format_epgm = True
            self.g_epgm = EPGM(path)
            graphs = self.g_epgm.G["graphs"]
            for g in graphs:
                if g["meta"]["label"] == dataset_name:
                    self.g_id = g["id"]

            self.g_nx = self.g_epgm.to_nx(self.g_id)
            g_vertices = self.g_epgm.G["vertices"]  # retrieve all graph vertices

            if node_type is None:
                node_type = self.g_epgm.node_types(self.g_id)
                if len(node_type) == 1:
                    node_type = node_type[0]
                else:
                    raise Exception(
                        "Multiple node types detected in graph {}: {}.".format(
                            self.g_id, node_type
                        )
                    )

            if target_attribute is None:
                target_attribute = self.g_epgm.node_attributes(self.g_id, node_type)
                if len(target_attribute) == 1:
                    target_attribute = target_attribute[0]
                else:
                    raise Exception(
                        "Multiple node attributes detected for nodes of type {} in graph {}: {}.".format(
                            node_type, self.g_id, target_attribute
                        )
                    )

            y = np.array(
                self._get_nodes(
                    g_vertices, node_type=node_type, target_attribute=target_attribute
                )
            )

        else:
            y_df = pd.read_csv(path, delimiter=" ", header=None, dtype=str)
            y_df.sort_values(by=[0], inplace=True)

            y = y_df.as_matrix()

        return y

    def _check_parameters(self, y, p, method, test_size, train_size, seed):
        """
        Checks that the parameters have valid values. It not, then it raises a ValueError exception with a
        message corresponding to the invalid parameter.

        Args:
            y: <numpy array> Array of size Nx2 containing node id, label columns.
            p: <int or float> Percent or count of the number of points for each class to sample.
            method: <str> One of 'count', 'percent', or 'absolute'.
            test_size: <int> number of points in the test set. For method 'count', it should be less than or equal to
            N - (np.unique(labels) * nc) where N is the number of labeled points in y.
            train_size: <int> The number of points in the train set only used by method 'absolute'.
            seed: <int> seed for random number generator, positive int or 0

        """
        if y is None:
            raise ValueError(
                "({}) y should be numpy array, not None".format(type(self).__name__)
            )
        if method != "count" and method != "percent" and method != "absolute":
            raise ValueError(
                "({}) Valid methods are 'count', 'percent', and 'absolute' not {}".format(
                    type(self).__name__, method
                )
            )

        if seed is not None:
            if seed < 0:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be positive integer or None.".format(
                        type(self).__name__
                    )
                )
            if type(seed) != int:
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )

        if method == "count":
            if type(p) != int or p <= 0:
                raise ValueError(
                    "({}) p should be positive integer".format(type(self).__name__)
                )
            if test_size is None or type(test_size) != int or test_size <= 0:
                raise ValueError(
                    "({}) test_size must be positive integer".format(
                        type(self).__name__
                    )
                )

        elif method == "percent":
            if type(p) != float or p < 0. or p > 1.:
                raise ValueError(
                    "({}) p should be float in the range [0,1].".format(
                        type(self).__name__
                    )
                )

        elif method == "absolute":
            if test_size is None or type(test_size) != int or test_size <= 0:
                raise ValueError(
                    "({}) test_size should be positive integer".format(
                        type(self).__name__
                    )
                )
            if train_size is None or type(train_size) != int or train_size <= 0:
                raise ValueError(
                    "({}) train_size should be positive integer".format(
                        type(self).__name__
                    )
                )

    def train_test_split(
        self, y=None, p=10, method="count", test_size=None, train_size=None, seed=None
    ):
        """
        Splits node data into train, test, validation, and unlabeled sets.

        Any points in y that have value UNKNOWN_TARGET_ATTRIBUTE are added to the unlabeled set.

        The validation set includes all the point that remain after the train, test and unlabeled sets have been
        created. As a result, it is possible the the validation set is empty, e.g., when method is set to 'percent'.

        The train, and test sets are build based on the specified method, 'count', 'percent', or 'absolute'.

        method='count': The value of parameter p specifies the number of points in the train set for each class. The
        test set size must be specified using the test_size parameter.

        method='percent': The value of parameter p specifies the train set size (and 1-p the test set size) as a
        percentage of the total number of points in y (including the unlabeled points.) The split is performed uniformly
        at random and the point labels (as specified in y) are not taken into account.

        method='absolute': The values of the parameters train_size and test_size specify the size of the train and test
        sets respectively. Points are selected uniformly at random and the label (as specified in y) are not taken into
        account.

        Args:
            y: <numpy array> Array of size Nx2 containing node id, label columns.
            p: <int or float> Percent or count of the number of points for each class to sample.
            method: <str> One of 'count', 'percent', or 'absolute'.
            test_size: <int> number of points in the test set. For method 'count', it should be less than or equal to
            N - (np.unique(labels) * nc) where N is the number of labeled points in y.
            train_size: <int> The number of points in the train set only used by method 'absolute'.
            seed: <int> seed for random number generator, positive int or 0

        Returns:
            y_train, y_val, y_test, y_unlabeled
        """
        self._check_parameters(
            y=y,
            p=p,
            method=method,
            test_size=test_size,
            train_size=train_size,
            seed=seed,
        )

        np.random.seed(seed=seed)

        if method == "count":
            return self._split_data(y, p, test_size)
        elif method == "percent":
            n_unlabelled_points = np.sum(y[:, 1] == UNKNOWN_TARGET_ATTRIBUTE)
            train_size = int((y.shape[0] - n_unlabelled_points) * p)
            test_size = y.shape[0] - n_unlabelled_points - train_size
            return self._split_data_absolute(
                y=y, test_size=test_size, train_size=train_size
            )
        elif method == "absolute":
            return self._split_data_absolute(
                y=y, test_size=test_size, train_size=train_size
            )

    def _split_data_absolute(self, y, test_size, train_size):
        """

        Args:
            y: <numpy.ndarray> Array of size N x 2 containing node id + labels.
            test_size: <int> number of points in test set.
            train_size: <int> The number of points in the train set.

        Returns:
            y_train, y_val, y_test, y_unlabeled

        """
        # The label column in y could include None type, that is point with no ground truth label. These, if any,will
        # be returned separately in y_unlabeled dataset
        y_used = np.zeros(y.shape[0])  # initialize all the points are available

        # indexes of points with no class label:
        ind = np.nonzero(y[:, 1] == UNKNOWN_TARGET_ATTRIBUTE)
        y_unlabeled = y[ind]
        y_used[ind] = 1

        ind = np.nonzero(y_used == 0)  # unused points
        ind_sampled = np.random.choice(ind[0], train_size, replace=False)
        y_train = y[ind_sampled]
        # mark these as used to make sure that they are not sampled for the test set
        y_used[ind_sampled] = 1

        # now sample test_size points for the test set
        ind = np.nonzero(y_used == 0)  # indexes of points that are not in training set
        if len(ind[0]) < test_size:
            raise Exception(
                "Not enough nodes available for the test set: available {} nodes, needed {}. Aborting".format(
                    len(ind[0]), test_size
                )
            )
        ind_sampled = np.random.choice(ind[0], test_size, replace=False)
        y_test = y[ind_sampled]
        y_used[ind_sampled] = 1
        # print("y_test shape: ", y_test.shape)

        # Validation set
        # the remaining labeled points (if any) go into the validation set
        ind = np.nonzero(y_used == 0)
        y_val = y[ind[0]]
        # print("y_val shape:", y_val.shape)

        return y_train, y_val, y_test, y_unlabeled

    def _split_data(self, y, nc, test_size):
        """
        Splits the data according to the scheme in Yang et al, ICML 2016, Revisiting semi-supervised learning with graph
        embeddings.

        :param y: <numpy.ndarray> Array of size N x 2 containing node id + labels.
        :param nc: <int> number of points from each class in train set.
        :param test_size: <int> number of points in test set; it should be less than or equal
        to N - (np.unique(labels) * nc).
        :return: y_train, y_val, y_test, y_unlabeled
        """

        # The label column in y could include None type, that is point with no ground truth label. These, if any,will
        # be returned separately in y_unlabeled dataset

        y_used = np.zeros(y.shape[0])  # initialize all the points are available

        # indexes of points with no class label:
        ind = np.nonzero(y[:, 1] == UNKNOWN_TARGET_ATTRIBUTE)
        y_unlabeled = y[ind]
        y_used[ind] = 1

        y_train = None

        class_labels = np.unique(y[:, 1])
        ind = class_labels == UNKNOWN_TARGET_ATTRIBUTE
        class_labels = class_labels[np.logical_not(ind)]

        if test_size > y.shape[0] - class_labels.size * nc:
            # re-adjust so that none of the training samples end up in the test set
            test_size = y.shape[0] - class_labels.size * nc

        for clabel in class_labels:
            # indexes of points with class label clabel:
            ind = np.nonzero(y[:, 1] == clabel)
            # select nc of these at random for the training set
            if ind[0].size <= nc:
                # too few labeled examples for class so use half for training and half for testing
                ind_selected = np.random.choice(ind[0], ind[0].size // 2, replace=False)
            else:
                ind_selected = np.random.choice(ind[0], nc, replace=False)
            # mark these as used to make sure that they are not sampled for the test set:
            y_used[ind_selected] = 1
            if y_train is None:
                y_train = y[ind_selected]
            else:
                # print("y_train shape:", y_train.shape)
                # print("y[ind_selected] shape:", y[ind_selected].shape)
                y_train = np.vstack((y_train, y[ind_selected]))

        # now sample test_size points for the test set
        ind = np.nonzero(y_used == 0)  # indexes of points that are not in training set
        if len(ind[0]) < test_size:
            raise Exception(
                "Not enough nodes available for the test set: available {} nodes, needed {}. Aborting".format(
                    len(ind[0]), test_size
                )
            )
        ind_selected = np.random.choice(ind[0], test_size, replace=False)
        y_test = y[ind_selected]
        y_used[ind_selected] = 1
        # print("y_test shape: ", y_test.shape)

        # the remaining points (if any) go into the validation set
        ind = np.nonzero(y_used == 0)
        y_val = y[ind[0]]
        # print("y_val shape:", y_val.shape)

        return y_train, y_val, y_test, y_unlabeled

    def write_data(self, output_dir, dataset_name, y_train, y_val, y_test, y_unlabeled):

        # if output directory does not exist, create it
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if self.format_epgm:
            return self._write_data_epgm(
                output_dir=output_dir,
                dataset_name=dataset_name,
                y_train=y_train,
                y_test=y_test,
                y_val=y_val,
                y_unlabeled=y_unlabeled,
            )
        else:
            return self._write_data(
                output_dir=output_dir,
                dataset_name=dataset_name,
                y_train=y_train,
                y_test=y_test,
                y_val=y_val,
                y_unlabeled=y_unlabeled,
            )

    def _write_data_epgm(
        self, output_dir, dataset_name, y_train, y_val, y_test, y_unlabeled
    ):

        G_train = self.g_nx.subgraph(y_train[:, 0]).copy()
        G_train.id = uuid.uuid4().hex
        G_train.name = dataset_name + "_train"

        G_val = self.g_nx.subgraph(y_val[:, 0]).copy()
        G_val.id = uuid.uuid4().hex
        G_val.name = dataset_name + "_val"

        G_test = self.g_nx.subgraph(y_test[:, 0]).copy()
        G_test.id = uuid.uuid4().hex
        G_test.name = dataset_name + "_test"

        G_unlabeled = self.g_nx.subgraph(y_unlabeled[:, 0]).copy()
        G_unlabeled.id = uuid.uuid4().hex
        G_unlabeled.name = dataset_name + "_unlabeled"

        # We are not interested in edges of G_train, G_val, and G_test, so remove them:
        G_train.remove_edges_from(list(G_train.edges()))
        G_val.remove_edges_from(list(G_val.edges()))
        G_test.remove_edges_from(list(G_test.edges()))
        G_unlabeled.remove_edges_from(list(G_unlabeled.edges()))

        self.g_epgm.append(G_train)
        self.g_epgm.append(G_val)
        self.g_epgm.append(G_test)
        self.g_epgm.append(G_unlabeled)

        # Finally write the graph to output_dir
        self.g_epgm.save(output_dir)

        return {"epgm_directory": output_dir}

    def _write_data(
        self, output_dir, dataset_name, y_train, y_val, y_test, y_unlabeled
    ):

        y_train_filename = os.path.join(output_dir, dataset_name + ".idx.train")
        np.savetxt(y_train_filename, y_train, fmt="%s")  # was fmt=%d

        y_val_filename = os.path.join(output_dir, dataset_name + ".idx.validation")
        np.savetxt(y_val_filename, y_val, fmt="%s")

        y_test_filename = os.path.join(output_dir, dataset_name + ".idx.test")
        np.savetxt(y_test_filename, y_test, fmt="%s")

        y_unlabeled_filename = os.path.join(output_dir, dataset_name + ".idx.unlabeled")
        np.savetxt(y_unlabeled_filename, y_unlabeled, fmt="%s")

        # return y_train_filename, y_val_filename, y_test_filename, y_unlabeled_filename
        return {
            "train_data_filename": y_train_filename,
            "val_data_filename": y_val_filename,
            "test_data_filename": y_test_filename,
            "unlabeled_data_filename": y_unlabeled_filename,
        }
