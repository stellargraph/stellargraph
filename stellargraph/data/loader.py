# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import warnings
import pandas as pd

import networkx as nx
from stellargraph.data.epgm import EPGM
from stellargraph.core.graph import *
from stellargraph import globalvar


def from_epgm(epgm_location, dataset_name=None, directed=False):
    """
    Imports a graph stored in EPGM format to a NetworkX object

    Args:
        epgm_location (str): The directory containing the EPGM data
        dataset_name (str), optional: The name of the dataset to import
        directed (bool): If True, load as a directed graph, otherwise
            load as an undirected graph

    Returns:
        A NetworkX graph containing the data for the EPGM-stored graph.
    """
    G_epgm = EPGM(epgm_location)
    graphs = G_epgm.G["graphs"]

    # if dataset_name is not given, use the name of the 1st graph head
    if not dataset_name:
        dataset_name = graphs[0]["meta"]["label"]
        warnings.warn(
            "dataset name not specified, using dataset '{}' in the 1st graph head".format(
                dataset_name
            ),
            RuntimeWarning,
        )

    # Select graph using dataset_name
    for g in graphs:
        if g["meta"]["label"] == dataset_name:
            graph_id = g["id"]

    # Convert to StellarGraph (via nx)
    Gnx = G_epgm.to_nx(graph_id, directed=directed)

    print(
        "Graph statistics: {} nodes, {} edges".format(
            Gnx.number_of_nodes(), Gnx.number_of_edges()
        )
    )
    return Gnx


def load_dataset_BlogCatalog3(location):
    """
    This method loads the BlogCatalog3 network dataset (http://socialcomputing.asu.edu/datasets/BlogCatalog3)
    into a networkx undirected heterogeneous graph.

    The graph has two types of nodes, 'user' and 'group', and two types of edges, 'friend' and 'belongs'.
    The 'friend' edges connect two 'user' nodes and the 'belongs' edges connects 'user' and 'group' nodes.

    The node and edge types are not included in the dataset that is a collection of node and group ids along with
    the list of edges in the graph.

    Important note about the node IDs: The dataset uses integers for node ids. However, the integers from 1 to 39 are
    used as IDs for both users and groups. This would cause a confusion when constructing the networkx graph object.
    As a result, we convert all IDs to string and append the character 'u' to the integer ID for user nodes and the
    character 'g' to the integer ID for group nodes.

    Args:
        location: <str> The directory where the dataset is located

    Returns:
        A networkx Graph object.

    """
    warnings.warn(
        "load_dataset_BlogCatalog3 has been replaced by `BlogCatalog3().load()`",
        DeprecationWarning,
    )
    from stellargraph.datasets import BlogCatalog3

    location = os.path.expanduser(location)
    return BlogCatalog3._load_from_location(location)
