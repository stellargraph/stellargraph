import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stellar.data.epgm import EPGM
from stellar.data.stellargraph import *


def from_epgm(
    epgm_location,
    dataset_name=None,
    directed=False,
    node_type_name="label",
    edge_type_name="label",
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
