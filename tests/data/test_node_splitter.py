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

import unittest
import uuid
import os
import numpy as np
import itertools as it
import networkx as nx
import pandas as pd

from stellargraph.data.stellargraph import StellarGraph
from stellargraph.data.node_splitter import NodeSplitter, train_val_test_split
from stellargraph.data.epgm import EPGM
from stellargraph import globals
from datetime import datetime, timedelta
import random


def create_heterogeneous_graph():
    g = nx.Graph()

    random.seed(42)  # produces the same graph every time

    start_date_dt = datetime.strptime("01/01/2015", "%d/%m/%Y")
    end_date_dt = datetime.strptime("01/01/2017", "%d/%m/%Y")
    start_end_days = (
        end_date_dt - start_date_dt
    ).days  # the number of days between start and end dates

    # 50 nodes of type person
    person_node_ids = list(range(0, 50))
    for person in person_node_ids:
        g.add_node(
            person, label="person", elite=random.choices(["0", "1", "-1"], k=1)[0]
        )

    # 200 nodes of type paper
    paper_node_ids = list(range(50, 250))
    g.add_nodes_from(paper_node_ids, label="paper")

    # 10 nodes of type venue
    venue_node_ids = list(range(250, 260))
    g.add_nodes_from(venue_node_ids, label="venue")

    # add the person - friend -> person edges
    # each person can be friends with 0 to 5 others; edges include a date
    for person_id in person_node_ids:
        k = random.randrange(5)
        friend_ids = set(random.sample(person_node_ids, k=k)) - {
            person_id
        }  # no self loops
        for friend in friend_ids:
            g.add_edge(
                person_id,
                friend,
                label="friend",
                date=(
                    start_date_dt + timedelta(days=random.randrange(start_end_days))
                ).strftime("%d/%m/%Y"),
            )

    # add the person - writes -> paper edges
    for person_id in person_node_ids:
        k = random.randrange(5)
        paper_ids = random.sample(paper_node_ids, k=k)
        for paper in paper_ids:
            g.add_edge(person_id, paper, label="writes")

    # add the paper - published-at -> venue edges
    for paper_id in paper_node_ids:
        venue_id = random.sample(venue_node_ids, k=1)[
            0
        ]  # paper is published at 1 venue only
        g.add_edge(paper_id, venue_id, label="published-at")

    return g


def delete_files_in_dir(path):
    for filename in os.listdir(path):
        filename_path = os.path.join(path, filename)
        if os.path.isfile(filename_path):
            os.unlink(filename_path)


def filter_nodes(nodes, node_type, target_attribute):
    """
    Returns a list of node IDs for the subset of graph_nodes that have the given node type.

    Args:
        nodes: <list> The node data as a list of tuples where for each node the first value is the node ID and the
        second value is a dictionary holding the node attribute data.
        node_type: <str> The node type to filter by
        target_attribute: <str> The target attribute key to filter by

    Returns:
        <list> List of 2-tuples where the first value is the node ID and the second value is the target attribute
        value.

    """
    # This code will fail if a node of node_type is missing the target_attribute.
    # We can fix this by using node['data'].get(target_attribute, None) so that at least all nodes of the
    # given type are returned. However, we must check for None in target_attribute later to exclude these nodes
    # from being added to train, test, and validation datasets.
    y = [
        (node[0], node[1].get(target_attribute, globals.UNKNOWN_TARGET_ATTRIBUTE))
        for node in nodes
        if node[1][globals.TYPE_ATTR_NAME] == node_type
    ]

    return y


def get_nodes(graph_nodes, node_type, target_attribute):
    """
    Returns a list of node IDs for the subset of graph_nodes that have the given node type.

    Args:
        graph_nodes: <list> List of OrderedDict with vertex data for graph in EPGM format
        node_type: <str> The node type of interest
        target_attribute: <str> The target attribute key

    Returns:
        List of 2-tuples where the first value is the node ID and the second value is the target attribute
        value.

    """
    # This code will fail if a node of node_type is missing the target_attribute.
    # We can fix this by using node['data'].get(target_attribute, None) so that at least all nodes of the
    # given type are returned. However, we must check for None in target_attribute later to exclude these nodes
    # from being added to train, test, and validation datasets.
    y = [
        (
            node["id"],
            node["data"].get(target_attribute, globals.UNKNOWN_TARGET_ATTRIBUTE),
        )
        for node in graph_nodes
        if node["meta"][globals.TYPE_ATTR_NAME] == node_type
    ]

    return y


def load_data(path, dataset_name=None, node_type=None, target_attribute=None):
    """
    Loads the node data

     :param path: Input filename or directory where graph in EPGM format is stored
     :param node_type: For HINs, the node type to consider
     :param target_attribute: For EPGM format, the target node attribute
     :return: N x 2 numpy arrays where the first column is the node id and the second column is the node label.
    """
    if os.path.isdir(path):
        g_epgm = EPGM(path)
        graphs = g_epgm.G["graphs"]
        for g in graphs:
            if g["meta"]["label"] == dataset_name:
                g_id = g["id"]

        g_vertices = g_epgm.G["vertices"]  # retrieve all graph vertices

        if node_type is None:
            node_type = g_epgm.node_types(g_id)
            if len(node_type) == 1:
                node_type = node_type[0]
            else:
                raise Exception(
                    "Multiple node types detected in graph {}: {}.".format(
                        g_id, node_type
                    )
                )

        if target_attribute is None:
            target_attribute = g_epgm.node_attributes(g_id, node_type)
            if len(target_attribute) == 1:
                target_attribute = target_attribute[0]
            else:
                raise Exception(
                    "Multiple node attributes detected for nodes of type {} in graph {}: {}.".format(
                        node_type, g_id, target_attribute
                    )
                )

        y = np.array(
            get_nodes(
                g_vertices, node_type=node_type, target_attribute=target_attribute
            )
        )

    else:
        y_df = pd.read_csv(path, delimiter=" ", header=None, dtype=str)
        y_df.sort_values(by=[0], inplace=True)

        y = y_df.values

    return y


class TestEPGMIOHeterogeneous(unittest.TestCase):
    def setUp(self):
        self.node_type = "person"
        self.target_attribute = "elite"
        self.ds_obj = NodeSplitter()

    def test_train_test_split_invalid_parameters(self):
        nc = 10
        test_size = 100
        method = "count"

        g = create_heterogeneous_graph()
        y = np.array(
            filter_nodes(
                list(g.nodes(data=True)),
                node_type=self.node_type,
                target_attribute=self.target_attribute,
            )
        )

        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=None,  # this will raise a ValueError exception
                p=nc,
                method=method,
                test_size=test_size,
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y,
                p=-1,
                method=method,
                test_size=test_size,  # this will raise a ValueError exception
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y,
                p=1.2,  # this will raise a ValueError exception
                method=method,
                test_size=test_size,
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y,
                p=0,
                method=method,
                test_size=test_size,  # this will raise a ValueError exception
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=nc, method=method, test_size=0
            )  # this will raise a ValueError exception
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=nc, method=method, test_size=-100
            )  # this will raise a ValueError exception
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=nc, method=method, test_size=99.10101
            )  # this will raise a ValueError exception
        # check parameter values for 'percent' method
        method = "percent"
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=1.0, method=method  # must be less than 1.
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=-0.5, method=method  # must be greater than or equalt 0.
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, p=10, method=method  # must be float in range (0, 1)
            )

        # check parameter values for 'absolute' method
        method = "absolute"
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, method=method, p=0.25
            )  # must specify train_size and test_size parameters, p is not used
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, method=method, test_size=0, train_size=1000
            )
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(y=y, method=method, test_size=99, train_size=0)
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y, method=method, test_size=0.25, train_size=0.75
            )  # test_size and train_size should be integers not percentages

        # test invalid method
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(
                y=y,
                method="other",  # valid values are 'percent', 'count', and 'absolute'
                p=nc,
                test_size=test_size,
            )
        # testing seed value
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(y=y, p=nc, test_size=100, seed=-1003)
        with self.assertRaises(ValueError):
            self.ds_obj.train_test_split(y=y, p=nc, test_size=100, seed=101.13)

    def test_split_data_epgm(self):

        nc = 5
        test_size = 20
        num_unlabeled = 14

        g = create_heterogeneous_graph()
        y = np.array(
            filter_nodes(
                list(g.nodes(data=True)),
                node_type=self.node_type,
                target_attribute=self.target_attribute,
            )
        )

        number_of_unique_labels = (
            len(np.unique(y[:, 1])) - 1
        )  # subtract one for missing value (-1) label

        validation_size = (
            y.shape[0] - test_size - nc * number_of_unique_labels - num_unlabeled
        )

        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=nc, test_size=test_size
        )

        self.assertEqual(
            self.y_test.shape[0],
            test_size,
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], test_size
            ),
        )

        self.assertEqual(
            self.y_train.shape[0],
            nc * number_of_unique_labels,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], nc * number_of_unique_labels
            ),
        )
        self.assertEqual(
            self.y_val.shape[0],
            validation_size,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_val.shape[0], validation_size
            ),
        )


class TestEPGMIOHomogenous(unittest.TestCase):
    def setUp(self):
        if os.getcwd().split("/")[-1] == "tests":
            self.input_dir = os.path.expanduser("resources/data/cora/cora.epgm")
            self.input_lab = os.path.expanduser("resources/data/cora/cora.lab/cora.lab")
        else:
            self.input_dir = os.path.expanduser("tests/resources/data/cora/cora.epgm")
            self.input_lab = os.path.expanduser(
                "tests/resources/data/cora/cora.lab/cora.lab"
            )

        self.dataset_name = "cora"
        self.node_type = "paper"
        self.target_attribute = "subject"
        self.ds_obj = NodeSplitter()

    def create_toy_dataset(self):
        # 100 node ids with 40 class 0, 40 class 1, and 20 unknown '-1'
        node_ids = [uuid.uuid4() for i in np.arange(100)]
        labels = ["-1"] * 100
        labels[0:40] = [0] * 40
        labels[40:80] = [1] * 40

        y = np.transpose(np.vstack((node_ids, labels)))

        return y

    def test_split_with_percent(self):
        method = "percent"
        p = 0.5

        y = self.create_toy_dataset()

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(
            y_train.shape,
            (40, 2),
            "Train set size is incorrect, expected (40, 2) but received {}".format(
                y_train.shape
            ),
        )
        self.assertEqual(
            y_test.shape,
            (40, 2),
            "Test set size is incorrect, expected (40, 2) but received {}".format(
                y_test.shape
            ),
        )
        self.assertEqual(
            y_unlabeled.shape,
            (20, 2),
            "Unlabeled set size is incorrect, expected (20, 2) but received {}".format(
                y_unlabeled.shape
            ),
        )
        self.assertEqual(
            y_val.shape,
            (0, 2),
            "Validation set size is incorrect, expected (0, 2) but received {}".format(
                y_val.shape
            ),
        )

        p = 0.33

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(
            y_train.shape,
            (26, 2),
            "Train set size is incorrect, expected (26, 2) but received {}".format(
                y_train.shape
            ),
        )
        self.assertEqual(
            y_test.shape,
            (54, 2),
            "Test set size is incorrect, expected (54, 2) but received {}".format(
                y_test.shape
            ),
        )
        self.assertEqual(
            y_unlabeled.shape,
            (20, 2),
            "Unlabeled set size is incorrect, expected (20, 2) but received {}".format(
                y_unlabeled.shape
            ),
        )
        self.assertEqual(
            y_val.shape,
            (0, 2),
            "Validation set size is incorrect, expected (0, 2) but received {}".format(
                y_val.shape
            ),
        )

        p = 0.75

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(
            y_train.shape,
            (60, 2),
            "Train set size is incorrect, expected (60, 2) but received {}".format(
                y_train.shape
            ),
        )
        self.assertEqual(
            y_test.shape,
            (20, 2),
            "Test set size is incorrect, expected (20, 2) but received {}".format(
                y_test.shape
            ),
        )
        self.assertEqual(
            y_unlabeled.shape,
            (20, 2),
            "Unlabeled set size is incorrect, expected (20, 2) but received {}".format(
                y_unlabeled.shape
            ),
        )
        self.assertEqual(
            y_val.shape,
            (0, 2),
            "Validation set size is incorrect, expected (0, 2) but received {}".format(
                y_val.shape
            ),
        )

        # remove points with UNKNOWN_TARGET_ATTRIBUTE
        y[80:, 1] = "2"

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(
            y_train.shape,
            (75, 2),
            "Train set size is incorrect, expected (75, 2) but received {}".format(
                y_train.shape
            ),
        )
        self.assertEqual(
            y_test.shape,
            (25, 2),
            "Test set size is incorrect, expected (25, 2) but received {}".format(
                y_test.shape
            ),
        )
        self.assertEqual(
            y_unlabeled.shape,
            (0, 2),
            "Unlabeled set size is incorrect, expected (0, 2) but received {}".format(
                y_unlabeled.shape
            ),
        )
        self.assertEqual(
            y_val.shape,
            (0, 2),
            "Validation set size is incorrect, expected (0, 2) but received {}".format(
                y_val.shape
            ),
        )

        p = 0.33
        y_train, y_val, y_test, y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(
            y_train.shape,
            (33, 2),
            "Train set size is incorrect, expected (33, 2) but received {}".format(
                y_train.shape
            ),
        )
        self.assertEqual(
            y_test.shape,
            (67, 2),
            "Test set siz   e is incorrect, expected (67, 2) but received {}".format(
                y_test.shape
            ),
        )
        self.assertEqual(
            y_unlabeled.shape,
            (0, 2),
            "Unlabeled set size is incorrect, expected (0, 2) but received {}".format(
                y_unlabeled.shape
            ),
        )
        self.assertEqual(
            y_val.shape,
            (0, 2),
            "Validation set size is incorrect, expected (0, 2) but received {}".format(
                y_val.shape
            ),
        )

    def test_split_data_lab(self):

        nc = 20
        test_size = 100
        method = "count"
        # this operation is also performed in test_load_epgm() but the call to setUp sets self.y to None so
        # I have to load the data again.
        y = load_data(
            self.input_lab,
            dataset_name=self.dataset_name,
            node_type=self.node_type,
            target_attribute=self.target_attribute,
        )

        number_of_unique_labels = len(np.unique(y[:, 1]))

        validation_size = y.shape[0] - test_size - nc * number_of_unique_labels
        #
        # Test using method 'count'
        #
        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=nc, test_size=test_size
        )

        self.assertEqual(
            self.y_test.shape[0],
            test_size,
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], test_size
            ),
        )

        self.assertEqual(
            self.y_train.shape[0],
            nc * number_of_unique_labels,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], nc * number_of_unique_labels
            ),
        )
        self.assertEqual(
            self.y_val.shape[0],
            validation_size,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_val.shape[0], validation_size
            ),
        )
        #
        # Test using method 'percent'
        #
        p = 0.75
        method = "percent"
        # y_val should be empty
        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(len(self.y_val), 0, "Validation set should be empty.")
        self.assertEqual(
            y.shape[0],
            self.y_train.shape[0] + self.y_test.shape[0] + self.y_unlabeled.shape[0],
            "The total number of points sampled is not equal to the size of y. Sampled {:d} vs expected {:d}".format(
                self.y_train.shape[0]
                + self.y_test.shape[0]
                + self.y_unlabeled.shape[0],
                y.shape[0],
            ),
        )

        self.assertEqual(
            self.y_train.shape[0],
            int(y.shape[0] * p),
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], int(y.shape[0] * p)
            ),
        )
        self.assertEqual(
            self.y_test.shape[0],
            int(y.shape[0] * (1. - p)),
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], int(y.shape[0] * (1. - p))
            ),
        )

        #
        # Test using method 'absolute'
        #
        method = "absolute"
        train_size = 1000
        test_size = 98
        # y_val should be empty
        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, method=method, test_size=test_size, train_size=train_size
        )
        validation_size = y.shape[0] - (
            train_size + test_size + self.y_unlabeled.shape[0]
        )
        self.assertEqual(
            self.y_val.shape[0], validation_size, "Validation set has incorrect size."
        )
        self.assertEqual(
            y.shape[0],
            self.y_train.shape[0]
            + self.y_test.shape[0]
            + self.y_unlabeled.shape[0]
            + self.y_val.shape[0],
            "The total number of points sampled is not equal to the size of y. Sampled {:d} vs expected {:d}".format(
                self.y_train.shape[0]
                + self.y_test.shape[0]
                + self.y_unlabeled.shape[0]
                + self.y_val.shape[0],
                y.shape[0],
            ),
        )
        self.assertEqual(
            self.y_train.shape[0],
            train_size,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], train_size
            ),
        )
        self.assertEqual(
            self.y_test.shape[0],
            test_size,
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], test_size
            ),
        )

    def test_split_data_epgm(self):

        nc = 20
        test_size = 100

        y = load_data(
            self.input_dir,
            dataset_name=self.dataset_name,
            node_type=self.node_type,
            target_attribute=self.target_attribute,
        )

        number_of_unique_labels = len(np.unique(y[:, 1]))

        validation_size = y.shape[0] - test_size - nc * number_of_unique_labels

        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=nc, test_size=test_size
        )

        self.assertEqual(
            self.y_test.shape[0],
            test_size,
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], test_size
            ),
        )

        self.assertEqual(
            self.y_train.shape[0],
            nc * number_of_unique_labels,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], nc * number_of_unique_labels
            ),
        )
        self.assertEqual(
            self.y_val.shape[0],
            validation_size,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_val.shape[0], validation_size
            ),
        )
        #
        # Test using method 'percent'
        #
        p = 0.5
        method = "percent"
        # y_val should be empty
        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, p=p, method=method
        )

        self.assertEqual(len(self.y_val), 0, "Validation set should be empty.")
        self.assertEqual(
            y.shape[0],
            self.y_train.shape[0] + self.y_test.shape[0] + self.y_unlabeled.shape[0],
            "The total number of points sampled is not equal to the size of y. Sampled {:d} vs expected {:d}".format(
                self.y_train.shape[0]
                + self.y_test.shape[0]
                + self.y_unlabeled.shape[0],
                y.shape[0],
            ),
        )

        self.assertEqual(
            self.y_train.shape[0],
            int(y.shape[0] * p),
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], int(y.shape[0] * p)
            ),
        )
        self.assertEqual(
            self.y_test.shape[0],
            int(y.shape[0] * (1. - p)),
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], int(y.shape[0] * (1. - p))
            ),
        )
        #
        # Test using method 'absolute'
        #
        method = "absolute"
        train_size = 299
        test_size = 101
        # y_val should be empty
        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.train_test_split(
            y=y, method=method, test_size=test_size, train_size=train_size
        )
        validation_size = y.shape[0] - (
            train_size + test_size + self.y_unlabeled.shape[0]
        )
        self.assertEqual(
            self.y_val.shape[0], validation_size, "Validation set has incorrect size."
        )
        self.assertEqual(
            y.shape[0],
            self.y_train.shape[0]
            + self.y_test.shape[0]
            + self.y_unlabeled.shape[0]
            + self.y_val.shape[0],
            "The total number of points sampled is not equal to the size of y. Sampled {:d} vs expected {:d}".format(
                self.y_train.shape[0]
                + self.y_test.shape[0]
                + self.y_unlabeled.shape[0]
                + self.y_val.shape[0],
                y.shape[0],
            ),
        )
        self.assertEqual(
            self.y_train.shape[0],
            train_size,
            "Train dataset has wrong size {:d} vs expected {:d}".format(
                self.y_train.shape[0], train_size
            ),
        )
        self.assertEqual(
            self.y_test.shape[0],
            test_size,
            "Test dataset has wrong size {:d} vs expected {:d}".format(
                self.y_test.shape[0], test_size
            ),
        )


##################
# Test the simple node_splitter interface:


def create_example_graph_1():
    sg = StellarGraph()
    sg.add_nodes_from([0, 1, 2, 3], label="movie")
    sg.add_nodes_from([4, 5, 6], label="person")
    sg.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, 3)], label="rating")
    sg.add_edges_from([(0, 4), (1, 4), (1, 5), (2, 4), (3, 5)], label="another")
    sg.add_edges_from([(4, 5)], label="friend")
    return sg


def create_example_graph_2():
    sg = StellarGraph()
    sg.add_nodes_from([0, 1, 2, "3", 4, 5, 6], label="default")
    sg.add_edges_from([(4, 0), (4, 1), (5, 1), (4, 2), (5, "3")], label="default")
    return sg


def test_split_function():
    # Example graph:
    for sg in [create_example_graph_1(), create_example_graph_2()]:

        # We have to have a target value for the nodes
        for n in sg:
            sg.node[n][sg._target_attr] = 1

        sg.fit_attribute_spec()

        splits = train_val_test_split(
            sg, node_type=None, test_size=2, train_size=3, stratify=False, seed=None
        )
        assert len(splits[0]) == 3
        assert len(splits[1]) == 2
        assert len(splits[2]) == 2
        assert len(splits[3]) == 0

        # Make sure the nodeIDs can be found in the graph
        assert all(s in sg for s in it.chain(*splits))


def test_split_function_percent():
    # Example graph:
    for sg in [create_example_graph_1(), create_example_graph_2()]:

        # We have to have a target value for the nodes
        for n in sg:
            sg.node[n][sg._target_attr] = 1

        # Test splits by proportion - note floor of the
        # number of samples
        splits = train_val_test_split(
            sg,
            node_type=None,
            test_size=2.8 / 7,
            train_size=3.2 / 7,
            stratify=False,
            seed=None,
        )

        # Note the length of val is still 2 even though we requested 1
        assert len(splits[0]) == 3
        assert len(splits[1]) == 2
        assert len(splits[2]) == 2
        assert len(splits[3]) == 0

        # Make sure the nodeIDs can be found in the graph
        assert all(s in sg for s in it.chain(*splits))


def test_split_function_stratify():
    # Example graph:
    sg = create_example_graph_2()

    # We have to have a target value for the nodes
    for ii, n in enumerate(sg):
        sg.node[n][sg._target_attr] = int(2 * ii / sg.number_of_nodes())

    splits = train_val_test_split(
        sg, node_type=None, test_size=2, train_size=4, stratify=True, seed=None
    )
    # For this number of nodes we should have 50% of the nodes as label 1
    assert sum(sg.node[s]["target"] for s in splits[0]) == len(splits[0]) // 2

    # This doesn't seem to be true for the test set though:
    # assert sum(sg.node[s]["target"] for s in splits[2]) == len(splits[2])//2

    # Make sure the nodeIDs can be found in the graph
    assert all(s in sg for s in it.chain(*splits))


def test_split_function_node_type():
    # Example graph:
    sg = create_example_graph_1()

    # We have to have a target value for the nodes
    for ii, n in enumerate(sg):
        sg.node[n][sg._target_attr] = int(2 * ii / sg.number_of_nodes())

    splits = train_val_test_split(
        sg, node_type="movie", test_size=1, train_size=2, stratify=False, seed=None
    )
    assert all(sg.node[s]["label"] == "movie" for split in splits for s in split)


def test_split_function_unlabelled():
    # Example graph:
    sg = create_example_graph_1()

    # Leave some of the nodes unlabelled:
    for ii, n in enumerate(sg):
        if ii > 2:
            sg.node[n][sg._target_attr] = 1

    splits = train_val_test_split(
        sg, node_type=None, test_size=2, train_size=2, stratify=False, seed=None
    )

    # For this number of nodes we should have 50% of the nodes as label 1
    # Note the length of val is still 2 even though we requested 1
    assert len(splits[0]) == 2
    assert len(splits[1]) == 0
    assert len(splits[2]) == 2
    assert len(splits[3]) == 3
