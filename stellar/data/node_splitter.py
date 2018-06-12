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
import logging
from stellar.data.epgm import EPGM

UNKNOWN_TARGET_ATTRIBUTE = '-1'


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
        y = [(node['id'], node['data'].get(target_attribute, UNKNOWN_TARGET_ATTRIBUTE))
             for node in graph_nodes if node['meta']['label'] == node_type]

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
            self.logger.debug('DATA SPLITTER: loading epgm graph from {}...'.format(path))
            self.format_epgm = True
            self.g_epgm = EPGM(path)
            graphs = self.g_epgm.G['graphs']
            for g in graphs:
                if g['meta']['label'] == dataset_name:
                    self.g_id = g['id']

            self.g_nx = self.g_epgm.to_nx(self.g_id)
            g_vertices = self.g_epgm.G['vertices']  # retrieve all graph vertices

            if node_type is None:
                node_type = self.g_epgm.node_types(self.g_id)
                if len(node_type) == 1:
                    node_type = node_type[0]
                    self.logger.info('target node type not specified, assuming {} node type'.format(node_type))
                else:
                    raise Exception('Multiple node types detected in graph {}: {}.'.format(self.g_id, node_type))

            if target_attribute is None:
                target_attribute = self.g_epgm.node_attributes(self.g_id, node_type)
                if len(target_attribute) == 1:
                    target_attribute = target_attribute[0]
                    self.logger.info('target node attribute not specified, assuming {} attribute'.format(target_attribute))
                else:
                    raise Exception(
                        'Multiple node attributes detected for nodes of type {} in graph {}: {}.'.format(node_type, self.g_id, target_attribute))

            y = np.array(self._get_nodes(g_vertices, node_type=node_type, target_attribute=target_attribute))

        else:
            self.logger.debug('DATA SPLITTER: loading indices and labels from {}...'.format(path))
            y_df = pd.read_csv(path, delimiter=' ', header=None, dtype=str)
            y_df.sort_values(by=[0], inplace=True)
            self.logger.debug("labels_df shape: {}".format(y_df.shape))

            y = y_df.as_matrix()

        self.logger.debug("labels_all shape: {}".format(y.shape))

        return y

    def split_data(self, y, nc, test_size):
        """
        Splits the data according to the scheme in Yang et al, ICML 2016, Revisiting semi-supervised learning with graph
        embeddings.

        Args:
            y (numpy.ndarray): Array of size N x 2 containing node id + labels.
            nc (int): number of points from each class in train set.
            test_size (int): number of points in test set;
                it should be less than or equal to N - (np.unique(labels) * nc).

        Returns:
            y_train, y_val, y_test, y_unlabeled
        """

        # The label column in y could include None type, that is point with no ground truth label. These, if any,will
        # be returned separately in y_unlabeled dataset

        y_used = np.zeros(y.shape[0])  # initialize all the points are available

        ind = np.nonzero(y[:, 1] == UNKNOWN_TARGET_ATTRIBUTE)  # indexes of points with no class lable
        y_unlabeled = y[ind]
        y_used[ind] = 1

        y_train = None
        self.logger.debug("nc = {} and test_size = {}".format(nc, test_size))

        class_labels = np.unique(y[:, 1])
        ind = class_labels == UNKNOWN_TARGET_ATTRIBUTE
        class_labels = class_labels[np.logical_not(ind)]

        if test_size > y.shape[0] - class_labels.size * nc:
            # re-adjust so that none of the training samples end up in the test set
            test_size = y.shape[0] - class_labels.size * nc

        for clabel in class_labels:
            ind = np.nonzero(y[:, 1] == clabel)  # indexes of points with class label clabel
            # select nc of these at random for the training set
            if ind[0].size <= nc:
                # too few labeled examples for class so use half for training and half for testing
                ind_selected = np.random.choice(ind[0], ind[0].size // 2, replace=False)
            else:
                ind_selected = np.random.choice(ind[0], nc, replace=False)
            y_used[ind_selected] = 1  # mark these as used to make sure that they are not sampled for the test set
            if y_train is None:
                y_train = y[ind_selected]
            else:
                # print("y_train shape:", y_train.shape)
                # print("y[ind_selected] shape:", y[ind_selected].shape)
                y_train = np.vstack((y_train, y[ind_selected]))

        # now sample test_size points for the test set
        ind = np.nonzero(y_used == 0)  # indexes of points that are not in training set
        if len(ind[0]) < test_size:
            raise Exception('Not enough nodes available for the test set: available {} nodes, needed {}. Aborting'.format(len(ind[0]), test_size))
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
            return self._write_data_epgm(output_dir=output_dir, dataset_name=dataset_name,
                                         y_train=y_train, y_test=y_test, y_val=y_val, y_unlabeled=y_unlabeled)
        else:
            return self._write_data(output_dir=output_dir, dataset_name=dataset_name,
                                    y_train=y_train, y_test=y_test, y_val=y_val, y_unlabeled=y_unlabeled)

    def _write_data_epgm(self, output_dir, dataset_name, y_train, y_val, y_test, y_unlabeled):

        G_train = self.g_nx.subgraph(y_train[:, 0]).copy()
        G_train.id = uuid.uuid4().hex
        G_train.name = dataset_name+'_train'

        G_val = self.g_nx.subgraph(y_val[:, 0]).copy()
        G_val.id = uuid.uuid4().hex
        G_val.name = dataset_name+'_val'

        G_test = self.g_nx.subgraph(y_test[:, 0]).copy()
        G_test.id = uuid.uuid4().hex
        G_test.name = dataset_name+'_test'

        G_unlabeled = self.g_nx.subgraph(y_unlabeled[:, 0]).copy()
        G_unlabeled.id = uuid.uuid4().hex
        G_unlabeled.name = dataset_name+'_unlabeled'

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

    def _write_data(self, output_dir, dataset_name, y_train, y_val, y_test, y_unlabeled):

        y_train_filename = os.path.join(output_dir, dataset_name+'.idx.train')
        self.logger.debug("Train data filename {}".format(y_train_filename))
        np.savetxt(y_train_filename, y_train, fmt='%s')  # was fmt=%d

        y_val_filename = os.path.join(output_dir, dataset_name+'.idx.validation')
        self.logger.debug("Validation data filename {}".format(y_val_filename))
        np.savetxt(y_val_filename, y_val, fmt='%s')

        y_test_filename = os.path.join(output_dir, dataset_name+'.idx.test')
        self.logger.debug("Test data filename {}".format(y_test_filename))
        np.savetxt(y_test_filename, y_test, fmt='%s')

        y_unlabeled_filename = os.path.join(output_dir, dataset_name+'.idx.unlabeled')
        self.logger.debug("Unlabeled data filename {}".format(y_unlabeled_filename))
        np.savetxt(y_unlabeled_filename, y_unlabeled, fmt='%s')

        #return y_train_filename, y_val_filename, y_test_filename, y_unlabeled_filename
        return {
            'train_data_filename': y_train_filename,
            'val_data_filename': y_val_filename,
            'test_data_filename': y_test_filename,
            'unlabeled_data_filename': y_unlabeled_filename,
        }