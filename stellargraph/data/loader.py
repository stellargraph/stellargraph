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
            stacklevel=2,
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
