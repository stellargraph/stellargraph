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

import pytest
import os
import numpy as np
import networkx as nx
from utils.edge_splitter import EdgeSplitter
from stellargraph.data.epgm import EPGM
from utils.node2vec_feature_learning import Node2VecFeatureLearning


def read_graph(graph_file, dataset_name, directed=False, weighted=False):
    """
    Reads the input network in networkx.

    :param graph_file: The directory where graph in EPGM format is stored
    :param dataset_name: The name of the graph selected out of all the graph heads in EPGM file
    :return: The graph in networkx format
    """
    try:   # assume args.input points to an EPGM graph
        G_epgm = EPGM(graph_file)
        graphs = G_epgm.G['graphs']
        if dataset_name is None:  # if dataset_name is not given, use the name of the 1st graph head
            dataset_name = graphs[0]['meta']['label']
            print('WARNING: dataset name not specified, using dataset \'{}\' in the 1st graph head'.format(dataset_name))
        graph_id = None
        for g in graphs:
            if g['meta']['label'] == dataset_name:
                graph_id = g['id']

        g = G_epgm.to_nx(graph_id, directed)
        if weighted:
            raise NotImplementedError
        else:
            # This is the correct way to set the edge weight in a MultiGraph.
            edge_weights = {e: 1 for e in g.edges(keys=True)}
            nx.set_edge_attributes(g, 'weight', edge_weights)
    except:   # otherwise, assume arg.input points to an edgelist file
        if weighted:
            g = nx.read_edgelist(graph_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            g = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.DiGraph())
            for edge in g.edges():
                g[edge[0]][edge[1]]['weight'] = 1

        if not directed:
            g = g.to_undirected()

    if not nx.is_connected(g):
        print("Graph is not connected")
        # take the largest connected component as the data
        g = max(nx.connected_component_subgraphs(g, copy=True), key=len)
        print('Largest subgraph statistics: {} nodes, {} edges'.format(g.number_of_nodes(),
                                                                       g.number_of_edges()))

    print('Graph statistics: {} nodes, {} edges'.format(g.number_of_nodes(), g.number_of_edges()))
    return g


class TestSelectMethodParameters(object):
    """
    Checks if the different methods throw the correct exception when given invalid parameter values
    """

    def test_fit(self):
        rl_obj = Node2VecFeatureLearning(nxG=None,
                                         embeddings_filename='')


        with pytest.raises(ValueError):
            rl_obj.fit(p=0)
            rl_obj.fit(p=-1)
            rl_obj.fit(p=1.1)
            rl_obj.fit(q=0)
            rl_obj.fit(q=-1)
            rl_obj.fit(q=1.5)
            rl_obj.fit(d=0)
            rl_obj.fit(d=-1)
            rl_obj.fit(d=100.5)
            rl_obj.fit(r=0)
            rl_obj.fit(r=-1)
            rl_obj.fit(r=5.001)
            rl_obj.fit(l=0)
            rl_obj.fit(l=-1)
            rl_obj.fit(l=0.1)
            rl_obj.fit(k=0)
            rl_obj.fit(k=-1)
            rl_obj.fit(k=9.009)

    def test_select_operator_from_str(self):
        """
        Checks that if the selected binary operator is not one of the valid operator, i.e., one of
        avg, l1, l2, h then it raises ValueError
        """
        rl_obj = Node2VecFeatureLearning(nxG=None,
                                         embeddings_filename='')

        with pytest.raises(ValueError):
            rl_obj.select_operator_from_str('other')
