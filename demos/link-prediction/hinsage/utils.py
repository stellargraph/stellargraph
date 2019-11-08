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

"""
Utility functions for the movielens-recommender demo
"""

from numba import jit
import numpy as np
import pandas as pd
import networkx as nx
import os


@jit(nopython=True)
def remap_ids(data, uid_map, mid_map, uid_inx=0, mid_inx=1):
    """
    Remap user and movie IDs
    """
    Nm = mid_map.shape[0]
    Nu = uid_map.shape[0]
    for ii in range(data.shape[0]):
        mid = data[ii, mid_inx]
        uid = data[ii, uid_inx]

        new_mid = np.searchsorted(mid_map, mid)
        new_uid = np.searchsorted(uid_map, uid)

        if new_mid < 0:
            print(mid, new_mid)

        # Only map to index if found, else map to zero
        if new_uid < Nu and (uid_map[new_uid] == uid):
            data[ii, uid_inx] = new_uid + Nm
        else:
            data[ii, uid_inx] = -1
        data[ii, mid_inx] = new_mid


def ingest_graph(data_path, config):
    """Ingest a graph from user-movie ratings"""
    edgelist_name = os.path.join(data_path, config["input_files"]["ratings"])

    columns = config["ratings_params"]["columns"]
    usecols = config["ratings_params"]["usecols"]
    sep = config["ratings_params"]["sep"]
    header = config["ratings_params"].get("header")

    # Load the edgelist:
    ratings = pd.read_csv(
        edgelist_name,
        names=columns,
        sep=sep,
        header=header,
        usecols=usecols,
        engine="python",
        dtype="int",
    )

    # Enumerate movies & users
    mids = np.unique(ratings["mId"])
    uids = np.unique(ratings["uId"])
    # Filter data and transform
    remap_ids(ratings.values, uids, mids)

    # Node ID map back to movie and user IDs
    movie_id_map = {i: "m_{}".format(mId) for i, mId in enumerate(mids)}
    user_id_map = {i + len(mids): "u_{}".format(uId) for i, uId in enumerate(uids)}
    id_map = {**movie_id_map, **user_id_map}
    inv_id_map = dict(zip(id_map.values(), id_map.keys()))

    # Create networkx graph
    g = nx.from_pandas_edgelist(
        ratings, source="uId", target="mId", edge_attr=True, create_using=nx.DiGraph()
    )

    # Add node types:
    node_types = {inv_id_map["m_" + str(v)]: "movie" for v in mids}
    node_types.update({inv_id_map["u_" + str(v)]: "user" for v in uids})

    nx.set_node_attributes(g, name="label", values=node_types)

    print(
        "Graph statistics: {} users, {} movies, {} ratings".format(
            sum([v[1]["label"] == "user" for v in g.nodes(data=True)]),
            sum([v[1]["label"] == "movie" for v in g.nodes(data=True)]),
            g.number_of_edges(),
        )
    )

    return g, id_map, inv_id_map


def ingest_features(data_path, config, node_type):
    """Ingest fatures for nodes of node_type"""
    filename = os.path.join(data_path, config["input_files"][node_type])

    if node_type == "users":
        parameters = config["user_feature_params"]
    elif node_type == "movies":
        parameters = config["movie_feature_params"]
    else:
        raise ValueError("Unknown node type {}".format(node_type))

    columns = parameters.get("columns")
    formats = parameters.get("formats")
    usecols = parameters.get("usecols")
    sep = parameters.get("sep", ",")
    feature_type = parameters.get("feature_type")
    dtype = parameters.get("dtype", "float32")
    header = parameters.get("header")

    # Load Data
    data = pd.read_csv(
        filename,
        index_col=0,
        names=columns,
        sep=sep,
        header=header,
        engine="python",
        usecols=usecols,
    )

    return data


def add_features_to_nodes(g, inv_id_map, user_features, movie_features):
    """Add user and movie features to graph nodes"""

    movie_features_dict = {
        k: np.array(movie_features.loc[k]) for k in movie_features.index
    }
    user_features_dict = {
        k: np.array(user_features.loc[k]) for k in user_features.index
    }

    node_features = {}
    for v in movie_features.index:
        node_features.update({inv_id_map["m_" + str(v)]: movie_features_dict[v]})

    for v in user_features.index:
        node_features.update({inv_id_map["u_" + str(v)]: user_features_dict[v]})

    nx.set_node_attributes(g, name="feature", values=node_features)

    return g
