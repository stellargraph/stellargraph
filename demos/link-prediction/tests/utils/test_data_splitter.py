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
import os
import numpy as np
from utils.data_splitter import DataSplitter
from stellar.data.epgm import EPGM
from shutil import rmtree


def delete_files_in_dir(path):
    for filename in os.listdir(path):
        filename_path = os.path.join(path, filename)
        if os.path.isfile(filename_path):
            os.unlink(filename_path)


class TestEPGMIOHeterogeneous(unittest.TestCase):

    def setUp(self):
        if os.getcwd().split('/')[-1] == 'tests':
            self.base_output_directory = os.path.expanduser('../../../tests/resources/data_splitter')
            self.input_dir = os.path.expanduser('../../../tests/resources/data/yelp/yelp.epgm')
            self.output_dir = os.path.expanduser('../../../tests/resources/data_splitter/yelp.epgm.out')
        else:
            self.base_output_directory = os.path.expanduser('../../tests/resources/data_splitter')
            self.input_dir = os.path.expanduser('../../tests/resources/data/yelp/yelp.epgm')
            self.output_dir = os.path.expanduser('../../tests/resources/data_splitter/yelp.epgm.out')

        self.dataset_name = 'small_yelp_example'
        self.node_type = 'user'
        self.target_attribute = 'elite'
        self.ds_obj = DataSplitter()

        if not os.path.isdir(self.base_output_directory):
            os.mkdir(self.base_output_directory)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # delete the files in the output directory
        delete_files_in_dir(self.output_dir)

    def tearDown(self):
        # delete the files in the output directory
        rmtree(self.base_output_directory)

    def test_load_epgm(self):

        y = self.ds_obj.load_data(self.input_dir,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)
        self.assertEqual(y.shape, (569, 2), "Did not load the correct number of node IDs and corresponding labels")

        # cora has 7 classes
        number_of_unique_labels = 3
        number_of_unique_labels_in_y = len(np.unique(y[:, 1]))

        self.assertEqual(number_of_unique_labels_in_y, number_of_unique_labels,
                         "Incorrect number of unique labels in y {:d} vs expected {:d}".format(number_of_unique_labels_in_y,
                                                                                               number_of_unique_labels))

    def test_split_data_epgm(self):

        nc = 10
        test_size = 100

        # this operation is also performed in test_load_epgm() but the call to setUp sets self.y to None so
        # I have to load the data again.
        y = self.ds_obj.load_data(self.input_dir,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        number_of_unique_labels = len(np.unique(y[:, 1])) - 1  # subtract one for missing value (-1) label

        validation_size = y.shape[0] - test_size - nc*number_of_unique_labels -10  # there are 10 unlabeled point in yelp

        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.assertEqual(self.y_test.shape[0], test_size,
                         "Test dataset has wrong size {:d} vs expected {:d}".format(self.y_test.shape[0], test_size))

        self.assertEqual(self.y_train.shape[0], nc*number_of_unique_labels,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_train.shape[0],
                                                                                     nc*number_of_unique_labels))
        self.assertEqual(self.y_val.shape[0], validation_size,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_val.shape[0],
                                                                                     validation_size))

    def test_write_data_epgm(self):

        nc = 10
        test_size = 100

        y = self.ds_obj.load_data(self.input_dir,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.ds_obj.write_data(self.output_dir,
                               dataset_name=self.dataset_name,
                               y_train=y_train,
                               y_test=y_test,
                               y_val=y_val,
                               y_unlabeled=y_unlabeled)

        # 3 files with json extension should be written in the output directory
        # The files are edges.json, graphs.json, vertices.json
        files_in_dir = os.listdir(self.output_dir)

        self.assertEqual(len(files_in_dir), 3,
                         "Incorrect number of .json files {:d} vs expected {:d}".format(len(files_in_dir), 3))

        self.assertTrue("edges.json" in files_in_dir, "Missing edges.json")
        self.assertTrue("vertices.json" in files_in_dir, "Missing vertices.json")
        self.assertTrue("graphs.json" in files_in_dir, "Missing graphs.json")

        # Load the EPGM and check it
        g_epgm = EPGM(self.output_dir)
        graphs = g_epgm.G['graphs']

        self.assertEqual(len(graphs), 5, "Incorrect number of graphs {:d} vs expected {:d}".format(len(graphs), 5))


class TestEPGMIOHomogenous(unittest.TestCase):

    def setUp(self):
        if os.getcwd().split('/')[-1] == 'tests':
            self.base_output_directory = os.path.expanduser('../../../tests/resources/data_splitter')
            self.input_dir = os.path.expanduser('../../../tests/resources/data/cora/cora.epgm')
            self.output_dir = os.path.expanduser('../../../tests/resources/data_splitter/cora.epgm.out')
            self.input_lab = os.path.expanduser('../../../tests/resources/data/cora/cora.lab/cora.lab')
            self.output_dir_lab = os.path.expanduser('../../../tests/resources/data_splitter/cora.out')
        else:
            self.base_output_directory = os.path.expanduser('../../tests/resources/data_splitter')
            self.input_dir = os.path.expanduser('../../tests/resources/data/cora/cora.epgm')
            self.output_dir = os.path.expanduser('../../tests/resources/data_splitter/cora.epgm.out')
            self.input_lab = os.path.expanduser('../../tests/resources/data/cora/cora.lab/cora.lab')
            self.output_dir_lab = os.path.expanduser('../../tests/resources/data_splitter/cora.out')

        if not os.path.isdir(self.base_output_directory):
            os.mkdir(self.base_output_directory)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.output_dir_lab):
            os.mkdir(self.output_dir_lab)

        self.dataset_name = 'cora'
        self.node_type = 'paper'
        self.target_attribute = 'subject'
        self.ds_obj = DataSplitter()

        # delete the contents of the output directories
        delete_files_in_dir(self.output_dir_lab)
        delete_files_in_dir(self.output_dir)

    def tearDown(self):
        # delete the contents of the output directories
        rmtree(self.base_output_directory)

    def test_load_lab(self):

        y = self.ds_obj.load_data(self.input_lab,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)
        self.assertEqual(y.shape, (2708, 2), "Did not load the correct number of node IDs and corresponding labels")

    def test_split_data_lab(self):

        nc = 20
        test_size = 100

        # this operation is also performed in test_load_epgm() but the call to setUp sets self.y to None so
        # I have to load the data again.
        y = self.ds_obj.load_data(self.input_lab,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        number_of_unique_labels = len(np.unique(y[:, 1]))

        validation_size = y.shape[0] - test_size - nc*number_of_unique_labels

        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.assertEqual(self.y_test.shape[0], test_size,
                         "Test dataset has wrong size {:d} vs expected {:d}".format(self.y_test.shape[0], test_size))

        self.assertEqual(self.y_train.shape[0], nc*number_of_unique_labels,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_train.shape[0],
                                                                                     nc*number_of_unique_labels))
        self.assertEqual(self.y_val.shape[0], validation_size,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_val.shape[0],
                                                                                     validation_size))

    def test_write_data_lab(self):

        nc = 20
        test_size = 100

        y = self.ds_obj.load_data(self.input_lab,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.ds_obj.write_data(self.output_dir_lab,
                               dataset_name=self.dataset_name,
                               y_train=y_train,
                               y_test=y_test,
                               y_val=y_val,
                               y_unlabeled=y_unlabeled)

        # 3 files with json extension should be written in the output directory
        # The files are edges.json, graphs.json, vertices.json
        files_in_dir = os.listdir(self.output_dir_lab)

        self.assertEqual(len(files_in_dir), 4,
                         "Incorrect number of .idx.* files {:d} vs expected {:d}".format(len(files_in_dir), 4))

        self.assertTrue(self.dataset_name+'.idx.train' in files_in_dir, "Missing .idx.train")
        self.assertTrue(self.dataset_name+'.idx.test' in files_in_dir, "Missing .idx.test")
        self.assertTrue(self.dataset_name+'.idx.validation' in files_in_dir, "Missing .idx.validation")
        self.assertTrue(self.dataset_name+'.idx.unlabeled' in files_in_dir, "Missing .idx.unlabeled")



    # Testing with data I/O in EPGM format
    def test_load_epgm(self):

        y = self.ds_obj.load_data(self.input_lab,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)
        self.assertEqual(y.shape, (2708, 2), "Did not load the correct number of node IDs and corresponding labels")

        # cora has 7 classes
        number_of_unique_labels = 7
        number_of_unique_labels_in_y = len(np.unique(y[:, 1]))

        self.assertEqual(number_of_unique_labels_in_y, number_of_unique_labels,
                         "Incorrect number of unique labels in y {:d} vs expected {:d}".format(number_of_unique_labels_in_y,
                                                                                               number_of_unique_labels))

    def test_split_data_epgm(self):

        nc = 20
        test_size = 100

        # this operation is also performed in test_load_epgm() but the call to setUp sets self.y to None so
        # I have to load the data again.
        y = self.ds_obj.load_data(self.input_dir,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        number_of_unique_labels = len(np.unique(y[:, 1]))

        validation_size = y.shape[0] - test_size - nc*number_of_unique_labels

        self.y_train, self.y_val, self.y_test, self.y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.assertEqual(self.y_test.shape[0], test_size,
                         "Test dataset has wrong size {:d} vs expected {:d}".format(self.y_test.shape[0], test_size))

        self.assertEqual(self.y_train.shape[0], nc*number_of_unique_labels,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_train.shape[0],
                                                                                     nc*number_of_unique_labels))
        self.assertEqual(self.y_val.shape[0], validation_size,
                         "Train dataset has wrong size {:d} vs expected {:d}".format(self.y_val.shape[0],
                                                                                     validation_size))

    def test_write_data_epgm(self):

        nc = 20
        test_size = 1000

        y = self.ds_obj.load_data(self.input_dir,
                                  dataset_name=self.dataset_name,
                                  node_type=self.node_type,
                                  target_attribute=self.target_attribute)

        y_train, y_val, y_test, y_unlabeled = self.ds_obj.split_data(y, nc=nc, test_size=test_size)

        self.ds_obj.write_data(self.output_dir,
                               dataset_name=self.dataset_name,
                               y_train=y_train,
                               y_test=y_test,
                               y_val=y_val,
                               y_unlabeled=y_unlabeled)

        # 3 files with json extension should be written in the output directory
        # The files are edges.json, graphs.json, vertices.json
        files_in_dir = os.listdir(self.output_dir)

        self.assertEqual(len(files_in_dir), 3,
                         "Incorrect number of .json files {:d} vs expected {:d}".format(len(files_in_dir), 3))

        self.assertTrue("edges.json" in files_in_dir, "Missing edges.json")
        self.assertTrue("vertices.json" in files_in_dir, "Missing vertices.json")
        self.assertTrue("graphs.json" in files_in_dir, "Missing graphs.json")

        # Load the EPGM and check it
        g_epgm = EPGM(self.output_dir)
        graphs = g_epgm.G['graphs']

        self.assertEqual(len(graphs), 5, "Incorrect number of graphs {:d} vs expected {:d}".format(len(graphs), 5))
