# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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
import networkx as nx
from stellargraph.data.epgm import EPGM


def read_graph(graph_file, dataset_name, is_directed=False, is_weighted=False):
    """
    Reads the input network in networkx.

    Args:
        graph_file: The directory where graph in EPGM format is stored.
        dataset_name: The name of the graph selected out of all the graph heads in EPGM file.

    Returns:
        The graph in networkx format
    """

    if graph_file.split(".")[-1] == "gpickle":
        g = nx.read_gpickle(graph_file)
        for edge in g.edges():
            g[edge[0]][edge[1]]["weight"] = 1  # {'weight': 1}

        if not is_directed:
            g = g.to_undirected()

        return g

    try:  # assume args.input points to an EPGM graph
        G_epgm = EPGM(graph_file)
        graphs = G_epgm.G["graphs"]
        if (
            dataset_name is None
        ):  # if dataset_name is not given, use the name of the 1st graph head
            dataset_name = graphs[0]["meta"]["label"]
            print(
                "WARNING: dataset name not specified, using dataset '{}' in the 1st graph head".format(
                    dataset_name
                )
            )
        graph_id = None
        for g in graphs:
            if g["meta"]["label"] == dataset_name:
                graph_id = g["id"]

        g = G_epgm.to_nx(graph_id, is_directed)
        if is_weighted:
            raise NotImplementedError
        else:
            # This is the correct way to set the edge weight in a MultiGraph.
            edge_weights = {e: 1 for e in g.edges(keys=True)}
            nx.set_edge_attributes(g, name="weight", values=edge_weights)
    except:  # otherwise, assume arg.input points to an edgelist file
        if is_weighted:
            g = nx.read_edgelist(
                graph_file,
                nodetype=int,
                data=(("weight", float),),
                create_using=nx.DiGraph(),
            )
        else:
            g = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.DiGraph())
            for edge in g.edges():
                g[edge[0]][edge[1]]["weight"] = 1  # {'weight': 1}

        if not is_directed:
            g = g.to_undirected()

    print(
        "Graph statistics: {} nodes, {} edges".format(
            g.number_of_nodes(), g.number_of_edges()
        )
    )
    return g
