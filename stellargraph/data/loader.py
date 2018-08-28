# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
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
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stellargraph.data.epgm import EPGM
from stellargraph.data.stellargraph import *
from stellargraph import globals

def from_epgm(
    epgm_location,
    dataset_name=None,
    directed=False,
    node_type_name=globals.TYPE_ATTR_NAME,
    edge_type_name=globals.TYPE_ATTR_NAME,
):
    """

    Args:
        epgm_location:
        dataset_name:
        graph_id:

    Returns:

    """
    G_epgm = EPGM(epgm_location)
    graphs = G_epgm.G["graphs"]

    # if dataset_name is not given, use the name of the 1st graph head
    if not dataset_name:
        dataset_name = graphs[0]["meta"]["label"]
        print(
            "WARNING: dataset name not specified, using dataset '{}' in the 1st graph head".format(
                dataset_name
            )
        )

    # Select graph using dataset_name
    for g in graphs:
        if g["meta"]["label"] == dataset_name:
            graph_id = g["id"]

    # Convert to StellarGraph (via nx)
    Gnx = G_epgm.to_nx(graph_id, directed=directed)

    if directed:
        G = StellarDiGraph(
            Gnx, node_type_name=node_type_name, edge_type_name=edge_type_name
        )
    else:
        G = StellarGraph(
            Gnx, node_type_name=node_type_name, edge_type_name=edge_type_name
        )

    print(
        "Graph statistics: {} nodes, {} edges".format(
            G.number_of_nodes(), G.number_of_edges()
        )
    )
    return G
