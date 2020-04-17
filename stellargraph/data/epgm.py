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

from collections import OrderedDict
import networkx as nx
from networkx.readwrite import json_graph
import os
import json
import uuid
import numpy as np
import chardet
import scipy.sparse as sp

import multiprocessing
from multiprocessing import Pool
from functools import partial

# from progressbar import ProgressBar, SimpleProgress
from time import sleep


def node_neighbours(v, edges):
    """Returns a list of neighbours of vertex v"""
    return (v, [e[1] for e in edges if e[0] == v])


def node_neighbours_extended(v, nodes, edges):
    """Returns a list of neighbours of vertex v"""
    nodes = np.array(nodes)
    edges = np.array(edges)
    mask = np.in1d(edges[:, 0], v)
    return (v, np.where(np.in1d(nodes, edges[mask, 1])), edges[mask, 1])


class EPGM(object):
    """EPGM class with converter methods to edgelist, adjacency matrix, etc."""

    def _progress(self, item_name, n, step, arg, i):
        """Display progress of a loop, e.g., list or dict comprehension statement"""
        if (i + 1) % step == 0:
            print(
                "{}> {} {} out of {} processed ({}%)".format(
                    "-" * int((i / n) * 80),
                    item_name,
                    i + 1,
                    n,
                    round((i + 1) / n * 100, 1),
                )
            )
        return arg

    def _progressbarUpdate(self, pbar, step, arg, i):
        """update progress bar"""
        if (i + 1) % step == 0:
            pbar.update(i + 1)
        return arg

    @classmethod
    def _nx_to_json(self, G):
        # Convert G to json:
        G_json = json_graph.node_link_data(G)
        G_json["graph"].update({"id": G.id})

        # Convert G_json['nodes'] to strings:
        for n in G_json["nodes"]:
            n["id"] = str(n["id"])

        # Fix source and target of G_json['links'] to actual node ids:
        for l in G_json["links"]:
            l["source"] = str(G_json["nodes"][l["source"]]["id"])
            l["target"] = str(G_json["nodes"][l["target"]]["id"])

        return G_json

    @classmethod
    def _json_to_epgm(self, G_json, node_attributes, node_labels):
        """Create G_epmg from G_json"""

        G_epgm = {
            "graphs": [OrderedDict()],
            "vertices": [OrderedDict(n) for n in G_json["nodes"]],
            "edges": [OrderedDict(e) for e in G_json["links"]],
        }

        # update G_epgm['graphs']:
        G_epgm["graphs"][0].update(
            OrderedDict(
                [
                    ("id", G_json["graph"]["id"]),
                    ("data", {}),
                    ("meta", {"label": G_json["graph"].get("name", "unnamed")}),
                ]
            )
        )

        # update G_epgm['vertices']:
        if node_labels is None:
            node_labels = [""] * len(G_epgm["vertices"])
        for ind, v in enumerate(G_epgm["vertices"]):
            if node_attributes is not None:
                data = {
                    str(k): str(v)
                    for k, v in node_attributes.iloc[ind].items()
                    if v != 0
                }  # extract non-zero node attributes (sparse node attributes)
            else:
                data = {}
            v.update(
                OrderedDict(
                    [
                        ("data", data),
                        (
                            "meta",
                            {
                                "label": str(node_labels[ind]),
                                "graphs": [G_json["graph"]["id"]],
                            },
                        ),
                    ]
                )
            )

        # update G_epgm['edges']:
        edges_key_order = ["id", "source", "target", "data", "meta"]
        for e in G_epgm["edges"]:
            e.update(
                OrderedDict(
                    [
                        ("id", uuid.uuid4().hex),
                        ("data", {}),
                        ("meta", {"label": "", "graphs": [G_json["graph"]["id"]]}),
                    ]
                )
            )
        # reorder keys in e to edges_key_order:
        G_epgm["edges"] = [
            OrderedDict((k, e[k]) for k in edges_key_order) for e in G_epgm["edges"]
        ]

        return G_epgm

    @classmethod
    def _reorder_keys(self, G):
        """Apply correct order of keys in self.G"""

        # Graphs:
        graphs_key_order = ["id", "data", "meta"]
        # reorder keys in g to graphs_key_order:
        G["graphs"] = [
            OrderedDict((k, g[k]) for k in graphs_key_order) for g in G["graphs"]
        ]

        # Vertices:
        vertices_key_order = ["id", "data", "meta"]
        # reorder keys in v to vertices_key_order:
        G["vertices"] = [
            OrderedDict((k, v[k]) for k in vertices_key_order) for v in G["vertices"]
        ]

        # Edges:
        edges_key_order = ["id", "source", "target", "data", "meta"]
        # reorder keys in e to edges_key_order:
        G["edges"] = [
            OrderedDict((k, e[k]) for k in edges_key_order) for e in G["edges"]
        ]

        return G

    @classmethod
    def load(self, path):
        """Load graphs from EPGM path"""
        if not os.path.isdir(path):
            raise Exception("Path {} does not exist!".format(path))

        G_epgm = {"graphs": list(), "vertices": list(), "edges": list()}

        for k in G_epgm.keys():
            # detect the codec:
            fname = os.path.join(path, str(k) + ".json")
            # with open(fname, 'rb') as fb:  # open the file for reading in binary format
            #     line = fb.readline()   # read the 1st line and evalulate its length
            #     fb.seek(0)   # return to the start of the file
            #     enc = chardet.detect(fb.read(1000*len(line)))['encoding']
            # detect the encoding from up to 1000 first lines in the file

            enc = "utf-8"  # just use 'utf-8' for all files
            with open(fname, "r", encoding=enc) as fp:
                print("...reading {} using {} encoding...".format(fp.name, enc))
                lines = fp.readlines()

            if isinstance(G_epgm[k], list):
                for l in lines:
                    G_epgm[k].append(json.loads(l))
            else:
                raise Exception(
                    "type(G[",
                    k,
                    "]): unexpected type",
                    type(G_epgm[k]).__name__,
                    ", stopping.",
                )

        G_epgm = self._reorder_keys(G=G_epgm)
        return G_epgm

    def node_types(self, graph_id):
        """List node labels(types) in graph with graph_id"""
        if not any([graph_id in g["id"] for g in self.G["graphs"]]):
            raise Exception("Graph with id {} does not exist".format(graph_id))

        node_types = [
            v["meta"]["label"]
            for v in self.G["vertices"]
            if graph_id in v["meta"]["graphs"]
        ]
        return np.unique(node_types)

    def node_attributes(self, graph_id, node_type):
        """Return a list of node attribute names, for nodes of node_type type, that belong to graph with graph_id"""
        x_ind = []
        nodes = [
            v
            for v in self.G["vertices"]
            if (graph_id in v["meta"]["graphs"]) and (node_type in v["meta"]["label"])
        ]
        for v in nodes:
            for k in list(v["data"].keys()):
                x_ind.append(k)

        return np.unique(x_ind)

    def node_attr_dim(self, graph_id, node_type):
        """Return the dimensionality of node attributes, for nodes of node_type type, that belong to graph with graph_id"""
        return len(np.unique(self.node_attributes(graph_id, node_type)))

    def __init__(self, G, node_attributes=None, node_labels=None):
        if "networkx.classes.graph.Graph" in str(G.__class__):
            G_json = self._nx_to_json(G)
            self.G = self._json_to_epgm(G_json, node_attributes, node_labels)
        elif "str" in str(G.__class__):  # this assumes that G is the path to EPGM graph
            self.G = self.load(G)
        else:
            raise Exception("G has unknown class ", str(G.__class__))

        self.G_nx = (
            {}
        )  # placeholder for storing graphs in networkx format (populated by calling the .to_nx() method)

    def append(self, G_add):
        """Update self.G by adding a new graph G_json_add"""
        G_json_add = self._nx_to_json(G_add)
        if any([G_json_add["graph"]["id"] in g["id"] for g in self.G["graphs"]]):
            raise Exception(
                "Graph with id {} already exists".format(G_json_add["graph"]["id"])
            )

        # update self.G['graphs']:
        G_epgm_add = self._json_to_epgm(G_json_add, None, None)  # was None, None, None
        self.G["graphs"].append(G_epgm_add["graphs"][0])

        # update self.G['vertices']:
        G_vertices = set([v["id"] for v in self.G["vertices"]])
        G_add_vertices = set([v["id"] for v in G_epgm_add["vertices"]])
        common_vertices = list(G_vertices.intersection(G_add_vertices))
        new_vertices = list(G_add_vertices.difference(G_vertices))

        # update common vertices in self.G and G_epgm_add:
        for v_id in common_vertices:
            i = [v["id"] == v_id for v in self.G["vertices"]].index(
                True
            )  # find the vertex index in self.G
            self.G["vertices"][i]["meta"]["graphs"].append(
                G_epgm_add["graphs"][0]["id"]
            )

        # append new vertices to self.G['vertices']:
        for v_id in new_vertices:
            i = [v["id"] == v_id for v in G_epgm_add["vertices"]].index(
                True
            )  # find the index of v_id in G_epgm_add['vertices']
            self.G["vertices"].append(
                G_epgm_add["vertices"][i]
            )  # add the vertex v_id to self.G['vertices']

        # update self.G['edges']:
        G_edges = set([(e["source"], e["target"]) for e in self.G["edges"]])
        G_add_edges = set([(e["source"], e["target"]) for e in G_epgm_add["edges"]])
        common_edges = list(G_edges.intersection(G_add_edges))
        new_edges = list(G_add_edges.difference(G_edges))

        # update common edges in self.G and G_epgm_add:
        for e_id in common_edges:
            i = [(e["source"], e["target"]) == e_id for e in self.G["edges"]].index(
                True
            )  # find the edge index in self.G
            self.G["edges"][i]["meta"]["graphs"].append(G_epgm_add["graphs"][0]["id"])

        # append new edges to self.G['edges']:
        for e_id in new_edges:
            i = [(e["source"], e["target"]) == e_id for e in G_epgm_add["edges"]].index(
                True
            )  # find the index of e_id in G_epgm_add['edges']
            self.G["edges"].append(
                G_epgm_add["edges"][i]
            )  # add the vertex e_id to self.G['edges']

    def to_nx_OLD(
        self,
        graph_id,
        directed=False,
        parallel_processing=True,
        n_jobs=multiprocessing.cpu_count(),
        progress=True,
        chunksize=100,
    ):
        """Convert the graph specified by its graph_id to networkx graph"""

        if (
            graph_id in self.G_nx.keys()
        ):  # if self.G_nx[graph_id] already exists, just return it, otherwise evaluate it
            return self.G_nx[graph_id]
        else:
            print("Converting the EPGM graph {} to NetworkX graph...".format(graph_id))
            if not any([graph_id in g["id"] for g in self.G["graphs"]]):
                raise Exception("Graph with id {} does not exist".format(graph_id))

            # List relevant nodes and edges:
            print("...extracting relevant nodes...", end="")
            nodes = [
                v["id"] for v in self.G["vertices"] if graph_id in v["meta"]["graphs"]
            ]
            print(" ...{} nodes extracted...".format(len(nodes)))
            print("...extracting relevant edges...", end="")
            edges = [
                (e["source"], e["target"])
                for e in self.G["edges"]
                if graph_id in e["meta"]["graphs"]
            ]
            print(" ...{} edges extracted...".format(len(edges)))
            # TODO: implement the case of weighted edges

            # create a graph as dict of lists in the format (node_id: [neighbour nodes])
            print("...building the graph as dict of lists...")
            print(
                "...[parallel_processing: {}, n_jobs: {}, progress_bar: {}]".format(
                    parallel_processing, n_jobs, progress
                )
            )

            if parallel_processing:  # parallel execution
                pool = Pool(processes=n_jobs)
                if progress:
                    n = len(nodes)
                    self.G_nx[graph_id] = []

                    # pbar = ProgressBar(
                    #     widgets=[
                    #         SimpleProgress(
                    #             format="%(value_s)s of %(max_value_s)s nodes processed (%(percentage)3d%%)"
                    #         )
                    #     ],
                    #     maxval=n,
                    # ).start()
                    # _ = [pool.apply_async(partial(node_neighbours, edges=edges), args=(v,),
                    #                       callback=self.G_nx[graph_id].append) for v in nodes]
                    # it seems that appending results using callback works much slower than either pool.map_async or pool.map
                    # while len(self.G_nx[graph_id]) != n:
                    #     pbar.update(len(self.G_nx[graph_id]))
                    #     sleep(1)

                    graph = pool.imap(
                        partial(node_neighbours, edges=edges), nodes, chunksize
                    )  # lazy map
                    # evaluate batches of imap, as the progress bar is being updated:
                    while len(self.G_nx[graph_id]) != n:
                        self.G_nx[graph_id].append(next(graph))
                        # pbar.update(len(self.G_nx[graph_id]))

                    # pbar.finish()

                    self.G_nx[graph_id] = dict(self.G_nx[graph_id])

                else:
                    self.G_nx[graph_id] = dict(
                        pool.map(partial(node_neighbours, edges=edges), nodes)
                    )

                pool.close()
                pool.join()

            else:  # sequential execution
                self.G_nx[graph_id] = {
                    v: [e[1] for e in edges if e[0] == v] for v in nodes
                }  # this works ~2.5x faster (for cora dataset) than the above for loop

            print("...converting the graph to nx format...")
            self.G_nx[graph_id] = nx.from_dict_of_lists(self.G_nx[graph_id])

            if directed:
                self.G_nx[graph_id] = self.G_nx[graph_id].to_directed()
            else:
                self.G_nx[graph_id] = self.G_nx[graph_id].to_undirected()

            return self.G_nx[graph_id]

    def to_nx(self, graph_id, directed=False, *args):
        """Convert the graph specified by its graph_id to networkx Directed Multi-graph"""

        if (
            False
        ):  # graph_id in self.G_nx.keys():  # if self.G_nx[graph_id] already exists, just return it, otherwise evaluate it
            return self.G_nx[graph_id]
        else:  # we always re-calculate self.G_nx, since directed argument can change
            print("Converting the EPGM graph {} to NetworkX graph...".format(graph_id))
            if not any([graph_id in g["id"] for g in self.G["graphs"]]):
                raise Exception("Graph with id {} does not exist".format(graph_id))

        self.G_nx[
            graph_id
        ] = (
            nx.MultiDiGraph()
        )  # create an empty directed graph that can store multiedges
        # add nodes to self.G_nx[graph_id], together with their attributes stored in 'data':
        self.G_nx[graph_id].add_nodes_from(
            [
                (v["id"], {**v["data"], **{"label": v["meta"].get("label", "")}})
                for v in self.G["vertices"]
            ]
        )
        # add edges to self.G_nx[graph_id], together with their attributes stored in 'data':
        # I have added the edge label in the edge data; sets the label to '' if the edges don't have a label
        self.G_nx[graph_id].add_edges_from(
            [
                (
                    e["source"],
                    e["target"],
                    e["id"],
                    {**e["data"], **{"label": e["meta"].get("label", "")}},
                )
                for e in self.G["edges"]
            ]
        )

        if not directed:
            self.G_nx[graph_id] = self.G_nx[graph_id].to_undirected()

        return self.G_nx[graph_id]

    def adjacency(self, graph_id, directed=False):
        """Return adjacency matrix of a graph specified by its graph_id"""

        if not any([graph_id in g["id"] for g in self.G["graphs"]]):
            raise Exception("Graph with id {} does not exist".format(graph_id))

        print("...building the adjacency matrix...")
        nodes = [v["id"] for v in self.G["vertices"] if graph_id in v["meta"]["graphs"]]
        adj = nx.adjacency_matrix(
            self.to_nx(graph_id, directed), nodelist=nodes
        )  # ensure the nodes in adj are ordered the same as in the epgm graph

        return adj

    # def adjacency_sans_nx(self, graph_id, directed=False):
    #     """Compose the adjacency matrix of a graph specified by its graph_id, NOT using networkx"""
    #     if not any([graph_id in g['id'] for g in self.G['graphs']]):
    #         raise Exception("Graph with id {} does not exist".format(graph_id))
    #
    #     # List relevant nodes and edges:
    #     nodes = [v['id'] for v in self.G['vertices'] if graph_id in v['meta']['graphs']]
    #     edges = [(e['source'], e['target']) for e in self.G['edges'] if graph_id in e['meta']['graphs']]
    #
    #     n_nodes = len(nodes)
    #     adj = sp.lil_matrix((n_nodes, n_nodes), dtype=int)  # TODO: implement the case of weighted edges
    #     for i, v in enumerate(nodes):
    #         data = data = partial(node_neighbours_extended, nodes=nodes, edges=edges)(v) # node_neighbours_extended(v, nodes, edges)
    #         nbr_idx = data[1][0]
    #         adj.rows[i] = list(nbr_idx)
    #         adj.data[i] = [1] * len(nbr_idx)
    #         # TODO: implement the case of weighted edges
    #
    #     if not directed:  # symmetrize the adj matrix
    #         adj = adj + adj.T - sp.diags(adj.diagonal(), dtype=int)
    #
    #     return adj

    # def adjacency_from_edgelist(self, graph_id, directed=False):
    #     """Return adjacency matrix of a graph specified by its graph_id"""
    #     # FIXME: the order of nodes in the created graph, and hence in adj, is different from that in the EPGM graph. Do not use this unless it's fixed!
    #     if not any([graph_id in g['id'] for g in self.G['graphs']]):
    #         raise Exception("Graph with id {} does not exist".format(graph_id))
    #
    #     # An alternative way, works faster, BUT GIVES A DIFFERENT NODES ORDER!!!
    #     # List relevant edges:
    #     edges = [(int(e['source']), int(e['target'])) for e in self.G['edges'] if graph_id in e['meta']['graphs']]
    #     # TODO: implement the case of weighted edges
    #     graph = nx.from_edgelist(edges)
    #     if directed:
    #         graph = graph.to_directed()
    #     else:
    #         graph = graph.to_undirected()
    #
    #     adj = nx.adjacency_matrix(graph)
    #
    #     return adj

    def edgelist(self, graph_id, directed=False):
        """Return edgelist of a graph specified by its graph_id"""

        print("...extracting the edgelist...")
        # edgelist = nx.to_edgelist(self.to_nx(graph_id, directed))
        edgelist = [
            (e["source"], e["target"])
            for e in self.G["edges"]
            if graph_id in e["meta"]["graphs"]
        ]  # works much faster, gets the edgelist directly from the edges.json part of epgm graph
        print(" ...{} edges extracted...".format(len(edgelist)))

        return edgelist

    def save(self, path):
        """
        Write self.G into three json files: graphs.json, vertices.json, and edges.json, in path directory
        """
        self.path = path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        for k in self.G.keys():
            with open(os.path.join(self.path, str(k) + ".json"), "w") as fp:
                if isinstance(self.G[k], list):
                    for l in self.G[k]:
                        json.dump(l, fp)
                        fp.write("\n")
                elif isinstance(self.G[k], OrderedDict) or isinstance(self.G[k], dict):
                    json.dump(self.G[k], fp)
                else:
                    print(
                        "type(G[",
                        k,
                        "]): unexpected type",
                        type(self.G[k]).__name__,
                        ", stopping.",
                    )
                    raise ()

    def save_as_graphml(self, graph_id, fname, directed):
        """
        Save the graph in GraphML XML format (e.g., for visualisation in gephi)
        Args:
            graph_id: unique id of the graph
            fname: file name to save the graphml into

        Returns:
            Exit code of nx.write_graphml()

        """
        if not any([graph_id in g["id"] for g in self.G["graphs"]]):
            raise Exception("Graph with id {} does not exist".format(graph_id))

        return nx.write_graphml(nx.DiGraph(self.to_nx(graph_id, directed)), path=fname)
