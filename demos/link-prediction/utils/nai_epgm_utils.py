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

'''
Utilities used by nai_scheduler.py and nai_scheduler_plugin.py
These unpack EPGM graphs to various files needed by the NAI pipelines, and save the resulting EPGM with predictions
'''

from utils.epgm import EPGM
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger('sciluigi-interface')

def build_vertex_map(vertices):
    """Build a map of vertex ids to vertex indices, and the inverse map"""
    vertex_map = {}
    inv_vertex_map = {}
    for i, v in enumerate(vertices):
        v_id = v['id']
        vertex_map[v_id] = i
        inv_vertex_map[i] = v_id

    return vertex_map, inv_vertex_map


def build_vertices_and_target_attribute(all_vertices, graph_id, node_type, target_attribute):

    selected_vertices = [v for v in all_vertices if graph_id in v['meta']['graphs'] and v['meta']['label'] == node_type]
    # The code below will fail if the target_attribute is missing from the data field of a relevant node
    # Fix by checking for missing attribute and setting target value to unknown
    vertices_and_targets = [(v['id'], v['data'].get(target_attribute, '-1')) for v in selected_vertices ]
    return selected_vertices, vertices_and_targets

def factorize(vertices_and_targets):

    vertex_labels = np.zeros(vertices_and_targets.shape[0], dtype=np.int)
    unique_labels = np.unique(vertices_and_targets[:, 1])

    if '-1' in unique_labels:

        vertex_labels[vertices_and_targets[:, 1] == '-1'] = -1

        ind = unique_labels == '-1'
        known_labels = unique_labels[np.logical_not(ind)]

        label_mapper = {v:k for k, v in enumerate(known_labels)}

        for class_label in known_labels:
            ind = vertices_and_targets[:, 1] == class_label
            vertex_labels[ind] = label_mapper[class_label]
    else:
        # how it was done before assuming no missing labels
        vertex_labels, unique_labels = pd.factorize(vertices_and_targets[:,1])

    return vertex_labels, unique_labels


def convert_from_EPGM(source_directory, dataset_name, output_directory, node_type=None, target_attribute=None,
                      attributes_to_ignore=None, write_to_disk = True):
    '''
    Reads a graph in EPGM format and writes to disk in "standard" format in the following files
        <datasetname>_edgelist.txt
        <datasetname>.att
        <datasetname>.lab
    :param source_directory: <str> The directory where the EPGM related files can be found.
    :param dataset_name: <str> The dataset name. It is used as prefix for the output files.
    :param output_directory: <str> The output directory
    :return: Graph as EPGM object, vertex maps from vertex ID to 0-based index, the inverse vertex map, and the
    unique vertex labels.
    '''

    # Load the EPGM graph:
    G_epgm = EPGM(source_directory)
    graphs = G_epgm.G['graphs']

    #  if there are more than one graph heads, then find the one with ['meta']['label'] == dataset_name
    graph_main_id = None
    for g in graphs:
        if g['meta']['label'] == dataset_name:
            graph_main_id = g['id']

    if graph_main_id is None:
        print("Could not find graph head name {:s}".format(dataset_name))
        raise Exception("Could not find graph with label {:s}".format(dataset_name))

    # Build a map (and inverse map) of vertex ids to vertex indices, for all vertices in the graph:
    all_vertices = G_epgm.G['vertices']
    vertex_map, inv_vertex_map = build_vertex_map(all_vertices)

    #  Extract the vertices and corresponding labels, for those vertices that belong to the graph with ID stored
    #  in graph_main_id, which have the required type:
    if node_type is None:
        node_type = G_epgm.node_types(graph_main_id)
        if len(node_type) == 1:
            node_type = node_type[0]
            print('target node type not specified, assuming {} node type'.format(node_type))
        else:
            raise Exception('Multiple node types detected in graph {}: {}.'.format(graph_main_id, node_type))

    if target_attribute is None:
        target_attribute = G_epgm.node_attributes(graph_main_id, node_type)
        if len(target_attribute) == 1:
            target_attribute = target_attribute[0]
            print('target node attribute not specified, assuming {} attribute'.format(target_attribute))
        else:
            raise Exception(
                'Multiple node attributes detected for nodes of type {} in graph {}: {}.'.format(node_type,
                                                                                                 graph_main_id,
                                                                                                 target_attribute))

    relevant_vertices, vertices_and_targets = build_vertices_and_target_attribute(all_vertices, graph_main_id,
                                                                                  node_type, target_attribute)
    print("Number of nodes in graph {} with node type {}: {}".format(graph_main_id, node_type, len(relevant_vertices)))

    #  Convert to N x 2 numpy array of string
    vertices_and_targets = np.array(vertices_and_targets)
    vertex_labels, unique_labels = factorize(vertices_and_targets)  # pd.factorize(vertices_and_targets[:, 1])
    vertices_and_targets[:, 1] = vertex_labels  # Turn string labels to integer factors

    #  Extract the node attributes to be used as predictors, if any
    if not hasattr(attributes_to_ignore, 'append'):
        attributes_to_ignore = []

    attributes_to_ignore.append(target_attribute)  # include the target attribute to the list of attributes to ignore
    predictors = sorted(set(G_epgm.node_attributes(graph_main_id, node_type)).difference(
        set(attributes_to_ignore)))  # use the rest of the attributes as predictors
    predictors_map = {p: i for i, p in enumerate(predictors)}
    num_predictors = len(predictors)

    if num_predictors > 0:
        num_vertices = len(relevant_vertices)
        predictors_array = np.zeros((num_vertices, num_predictors + 1), dtype=float)  # First column is vertex ID
        row = 0
        for v in relevant_vertices:
            predictors_array[row, 0] = vertex_map[v['id']]
            data = {att[0]: att[1] for att in v['data'].items() if
                    att[0] in predictors}  # dictionary with attributes
            for att_name, att_value in data.items():
                col = predictors_map[att_name]
                predictors_array[row, col + 1] = att_value
            row = row + 1

    # write the vertex label data to disk
    edge_list_filename = None
    vertex_labels_filename = None
    vertex_attributes_filename = None
    if write_to_disk:
        vertex_labels_filename = os.path.join(output_directory,dataset_name + '.lab')
        with open(vertex_labels_filename, 'w') as f:
            for v in vertices_and_targets:
                f.write("{:d} {:s}\n".format(vertex_map[v[0]], v[1]))

        if num_predictors > 0:
            # write the node attributes to disk
            vertex_attributes_filename = os.path.join(output_directory, dataset_name + '.att')
            np.savetxt(vertex_attributes_filename, predictors_array, '%f')

        # Extract the edgelist (ignoring the node types):
        edge_list = G_epgm.edgelist(graph_id=graph_main_id)

        # write the edge list to disk
        edge_list_filename = os.path.join(output_directory, dataset_name + '_edgelist.txt')
        with open(edge_list_filename, 'w') as f:
            for e in edge_list:
                f.write("{:d} {:d}\n".format(vertex_map[e[0]], vertex_map[e[1]]))

    return G_epgm, vertex_map, inv_vertex_map, unique_labels, edge_list_filename, vertex_labels_filename, vertex_attributes_filename


def write_to_epgm(input_epgm, temp_directory, output_epgm, g, inverse_vertex_map=None, unique_vertex_labels=None, target_attribute=None):
    '''
    Stores the graph with predicted vertex labels to disk in EPGM format.
    :param input_epgm: <str> The location of the input epgm file.
    :param temp_directory: <str> The directory where the vertex prediction will be read from a file with *.pred
    extension.
    :param output_epgm: <str> The output directory (relative to input_epgm directory)
    :param g: The EPGM object that holds the input graph.
    :return: None
    '''
    output_epgm_directory = os.path.join(input_epgm, output_epgm)
    if not os.path.exists(output_epgm_directory):
        os.mkdir(output_epgm_directory)

    # get list of files in temp_directory and find the one with *.pred extension
    all_files_in_temp = os.listdir(temp_directory)
    pred_filename = None
    for f in all_files_in_temp:
        if f.split('.')[-1] == 'pred':
            pred_filename = f
            break
    if pred_filename is None:
        print("Could not find .pred file in dir {:s}".format(temp_directory))

    if inverse_vertex_map is None or unique_vertex_labels is None:  # special case for GCN output
        # load the predictions
        vertex_label_predictions = np.loadtxt(os.path.join(temp_directory, pred_filename), dtype=str, delimiter=',')  # load as string
        # update the epgm graph
        g_vertices = g.G['vertices']
        for v_id, pred_lab in vertex_label_predictions:
            # find the corresponding vertex in g.G['vertices'] list
            for g_v in g_vertices:
                if g_v['id'] == v_id:  # str(inverse_vertex_map[v_id]):
                    # create a new entry in the dictionary where the predicted label will be stored.
                    # g_v['meta']['predicted label'] = vertex_labels[pred_lab]
                    g_v['data'].update({'PREDICTED_' + target_attribute : pred_lab}) #['PREDICTED_' + target_attribute] = vertex_labels[pred_lab]
                    break
    else:
        # load the predictions
        vertex_label_predictions = np.loadtxt(os.path.join(temp_directory, pred_filename), dtype=int)
        # update the epgm graph
        g_vertices = g.G['vertices']
        for v_id, pred_lab in vertex_label_predictions:
            # find the corresponding vertex in g.G['vertices'] list
            for g_v in g_vertices:
                if g_v['id'] == str(inverse_vertex_map[v_id]):
                    # create a new entry in the dictionary where the predicted label will be stored.
                    # g_v['meta']['predicted label'] = vertex_labels[pred_lab]
                    g_v['data'].update({'PREDICTED_' + target_attribute : unique_vertex_labels[pred_lab]}) #['PREDICTED_' + target_attribute] = vertex_labels[pred_lab]
                    break
    # write epgm to destination directory
    return g.save(output_epgm_directory)
