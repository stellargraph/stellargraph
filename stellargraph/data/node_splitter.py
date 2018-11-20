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

__all__ = ["train_val_test_split", "NodeSplitter"]

import numpy as np
import pandas as pd
from stellargraph.core.graph import StellarGraphBase
from stellargraph import globalvar


# Easier functional interface for the splitter:
def train_val_test_split(
    G,
    node_type=None,
    test_size=0.4,
    train_size=0.2,
    targets=None,
    split_equally=False,
    seed=None,
):
    """
        Splits node data into train, test, validation, and unlabeled sets.

        Any nodes that have a target value equal to globals.UNKNOWN_TARGET_ATTRIBUTE are added to the unlabeled set.

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

        targets: None or DataFrame or dict, optional (default=None)
            If False the nodes are randomly assigned to each partition. If this is a Pandas DataFrame
            with node_ids as the index, or a dictionary with node_ids as keys and target as values,
            then these values will be used to find unlabelled nodes and if `split_equally` is True
            these target values will be used to sample equal numbers of each class.

        split_equally: bool (default=False)
            if `split_equally` is True the values passed into the targets argument will be used to
            sample equal numbers for the train split by class label.

    Returns:
            y_train, y_val, y_test, y_unlabeled
    """
    node_splitter = NodeSplitter()

    # Get list of nodes to split
    if node_type is None:
        nodes = list(G)

    elif isinstance(G, StellarGraphBase):
        nodes = G.nodes_of_type(node_type)

    else:
        raise TypeError("G must be a StellarGraph is node_type is not None")

    # Number of nodes and number without a label
    n_nodes = len(nodes)

    # Extract the target information
    if targets is not None:
        # TODO: The equal sampling option will fail if these values are not hashable.
        # Check that split_equally_by_target is the correct type
        if isinstance(targets, pd.DataFrame):
            target_values = [
                targets.loc[n]
                if n in targets.index
                else globalvar.UNKNOWN_TARGET_ATTRIBUTE
                for n in nodes
            ]

        elif isinstance(targets, dict):
            target_values = [
                targets.get(n, globalvar.UNKNOWN_TARGET_ATTRIBUTE) for n in nodes
            ]

        else:
            raise TypeError(
                "The targets are expected to be either a Pandas DataFrame or a dict."
            )

        n_known = sum(t != globalvar.UNKNOWN_TARGET_ATTRIBUTE for t in targets)

    else:
        n_known = n_nodes
        target_values = [0] * n_nodes

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
    nodeid_and_label = np.array([nl for nl in enumerate(target_values)], dtype="U")

    # If stratified sampling, we need the target labels.
    if split_equally and targets is not None:
        class_set = set(target_values)

        # Remove the unknown target type
        class_set.discard(globalvar.UNKNOWN_TARGET_ATTRIBUTE)

        # The number of classes we have
        n_classes = len(class_set)

        if n_classes == 0:
            raise RuntimeError(
                "Found no usable target classes in split_equally_by_targets."
            )

        if train_size_n < n_classes:
            raise RuntimeError(
                "The number of classes must be smaller than the training size."
            )

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
        self._random = None

    def _get_nodes(self, graph_nodes, node_type, target_attribute):
        """
        Returns a list of node IDs for the subset of graph_nodes that have the given node type.

        Args:
            graph_nodes: <list> List of OrderedDict with vertex data for graph in EPGM format
            node_type: <str> The node type of interest
            target_attribute: <str> The target attribute key

        Returns:
            <list> List of node IDs that have given node type

        """
        # This code will fail if a node of node_type is missing the target_attribute.
        # We can fix this by using node['data'].get(target_attribute, None) so that at least all nodes of the
        # given type are returned. However, we must check for None in target_attribute later to exclude these nodes
        # from being added to train, test, and validation datasets.
        y = [
            (
                node["id"],
                node["data"].get(target_attribute, globalvar.UNKNOWN_TARGET_ATTRIBUTE),
            )
            for node in graph_nodes
            if node["meta"][globalvar.TYPE_ATTR_NAME] == node_type
        ]

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
        if not isinstance(y, np.ndarray):
            raise ValueError("({}) y should be numpy array".format(type(self).__name__))

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
            if not isinstance(seed, int):
                raise ValueError(
                    "({}) The random number generator seed value, seed, should be integer type or None.".format(
                        type(self).__name__
                    )
                )

        if method == "count":
            if not isinstance(p, int) or p <= 0:
                raise ValueError(
                    "({}) p should be positive integer".format(type(self).__name__)
                )
            if test_size is None or not isinstance(test_size, int) or test_size <= 0:
                raise ValueError(
                    "({}) test_size must be positive integer".format(
                        type(self).__name__
                    )
                )

        elif method == "percent":
            if not isinstance(p, float) or p < 0.0 or p > 1.0:
                raise ValueError(
                    "({}) p should be float in the range [0,1].".format(
                        type(self).__name__
                    )
                )

        elif method == "absolute":
            if test_size is None or not isinstance(test_size, int) or test_size <= 0:
                raise ValueError(
                    "({}) test_size should be positive integer".format(
                        type(self).__name__
                    )
                )
            if train_size is None or not isinstance(train_size, int) or train_size <= 0:
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

        Any points in y that have value globals.UNKNOWN_TARGET_ATTRIBUTE are added to the unlabeled set.

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

        if self._random is None:
            self._random = np.random.RandomState(seed=seed)

        if method == "count":
            return self._split_data(y, p, test_size)
        elif method == "percent":
            n_unlabelled_points = np.sum(y[:, 1] == globalvar.UNKNOWN_TARGET_ATTRIBUTE)
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
        Splits given data such that the sizes of the test and train sets are fixed to the values given.

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
        ind = np.nonzero(y[:, 1] == globalvar.UNKNOWN_TARGET_ATTRIBUTE)
        y_unlabeled = y[ind]
        y_used[ind] = 1

        ind = np.nonzero(y_used == 0)  # unused points
        ind_sampled = self._random.choice(ind[0], train_size, replace=False)
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
        ind_sampled = self._random.choice(ind[0], test_size, replace=False)
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

        Args:
            y: <numpy.ndarray> Array of size N x 2 containing node id + labels.
            nc: <int> number of points from each class in train set.
            test_size: <int> number of points in test set; it should be less than or equal to
            N - (np.unique(labels) * nc).

        Returns:
            y_train, y_val, y_test, y_unlabeled
        """
        y_used = np.zeros(y.shape[0])  # initialize all the points are available

        # indexes of points with no class label:
        ind = np.nonzero(y[:, 1] == globalvar.UNKNOWN_TARGET_ATTRIBUTE)
        y_unlabeled = y[ind]
        y_used[ind] = 1

        y_train = None

        class_labels = np.unique(y[:, 1])
        ind = class_labels == globalvar.UNKNOWN_TARGET_ATTRIBUTE
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
                ind_selected = self._random.choice(
                    ind[0], ind[0].size // 2, replace=False
                )
            else:
                ind_selected = self._random.choice(ind[0], nc, replace=False)
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
        ind_selected = self._random.choice(ind[0], test_size, replace=False)
        y_test = y[ind_selected]
        y_used[ind_selected] = 1
        # print("y_test shape: ", y_test.shape)

        # the remaining points (if any) go into the validation set
        ind = np.nonzero(y_used == 0)
        y_val = y[ind[0]]
        # print("y_val shape:", y_val.shape)

        return y_train, y_val, y_test, y_unlabeled
