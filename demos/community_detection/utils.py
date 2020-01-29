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

"""
Collection of utility functions related to the community detection. Mainly supports visualisation and data preparation.

"""

import pandas as pd
import numpy as np
import random
import networkx as nx
from sklearn.cluster import DBSCAN
from functools import reduce
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")  # supress warnings due to some future deprications


def load_features(input_data):
    # Summarise features by terrorist group
    dt_collect = input_data[
        ["eventid", "nperps", "success", "suicide", "nkill", "nwound", "gname"]
    ]
    dt_collect.fillna(0, inplace=True)
    dt_collect.nperps[dt_collect.nperps < 0] = 0

    summarize_by_gname = (
        dt_collect.groupby("gname")
        .agg(
            {
                "eventid": "count",
                "nperps": "sum",
                "nkill": "sum",
                "nwound": "sum",
                "success": "sum",
            }
        )
        .reset_index()
    )
    summarize_by_gname.columns = [
        "gname",
        "n_attacks",
        "n_nperp",
        "n_nkil",
        "n_nwound",
        "n_success",
    ]
    summarize_by_gname["success_ratio"] = (
        summarize_by_gname["n_success"] / summarize_by_gname["n_attacks"]
    )
    summarize_by_gname.drop(["n_success"], axis=1, inplace=True)

    # Collect counts of each attack type
    dt_collect = input_data[["gname", "attacktype1_txt"]]
    gname_attacktypes = (
        dt_collect.groupby(["gname", "attacktype1_txt"])["attacktype1_txt"]
        .count()
        .to_frame()
    )
    gname_attacktypes.columns = ["attacktype_count"]
    gname_attacktypes.reset_index(inplace=True)
    gname_attacktypes_wide = gname_attacktypes.pivot(
        index="gname", columns="attacktype1_txt", values="attacktype_count"
    )
    gname_attacktypes_wide.fillna(0, inplace=True)
    gname_attacktypes_wide.drop(["Unknown"], axis=1, inplace=True)

    # Collect counts of each target type
    dt_collect = input_data[["gname", "targtype1_txt"]]
    gname_targtypes = (
        dt_collect.groupby(["gname", "targtype1_txt"])["targtype1_txt"]
        .count()
        .to_frame()
    )
    gname_targtypes.columns = ["targtype_count"]
    gname_targtypes.reset_index(inplace=True)
    gname_targtypes_wide = gname_targtypes.pivot(
        index="gname", columns="targtype1_txt", values="targtype_count"
    )
    gname_targtypes_wide.fillna(0, inplace=True)
    gname_targtypes_wide.drop(["Unknown"], axis=1, inplace=True)

    # Combine all features
    data_frames = [summarize_by_gname, gname_attacktypes_wide, gname_targtypes_wide]
    gnames_features = reduce(
        lambda left, right: pd.merge(left, right, on=["gname"], how="outer"),
        data_frames,
    )
    return gnames_features


def load_network(input_data):
    # Create country_decade feature
    dt_collect = input_data[["eventid", "country_txt", "iyear", "gname"]]
    dt_collect["decade"] = (dt_collect["iyear"] // 10) * 10
    dt_collect["country_decade"] = (
        dt_collect["country_txt"] + "_" + dt_collect["decade"].map(str) + "s"
    )
    dt_collect = dt_collect[dt_collect.gname != "Unknown"]

    # Create a country_decade edgelist
    gnames_country_decade = (
        dt_collect.groupby(["gname", "country_decade"])
        .agg({"eventid": "count"})
        .reset_index()
    )
    gnames_country_decade_edgelist = pd.merge(
        gnames_country_decade, gnames_country_decade, on="country_decade", how="left"
    )
    gnames_country_decade_edgelist.drop(
        ["eventid_x", "eventid_y"], axis=1, inplace=True
    )
    gnames_country_decade_edgelist.columns = ["source", "country_decade", "target"]
    gnames_country_decade_edgelist = gnames_country_decade_edgelist[
        gnames_country_decade_edgelist.source != gnames_country_decade_edgelist.target
    ]

    G_country_decade = nx.from_pandas_edgelist(
        gnames_country_decade_edgelist, source="source", target="target"
    )

    # Create edgelist from the related column
    dt_collect = input_data["related"]
    dt_collect.dropna(inplace=True)
    gname_event_mapping = input_data[["eventid", "gname"]].drop_duplicates()
    gname_event_mapping.eventid = gname_event_mapping.eventid.astype(str)

    G_related = nx.parse_adjlist(
        dt_collect.values, delimiter=", "
    )  # attacks that are related
    df_related = nx.to_pandas_edgelist(G_related)
    df_related.replace(" ", "", regex=True, inplace=True)
    df_source_gname = pd.merge(
        df_related,
        gname_event_mapping,
        how="left",
        left_on="source",
        right_on="eventid",
    )
    df_source_gname.rename(columns={"gname": "gname_source"}, inplace=True)
    df_target_gname = pd.merge(
        df_source_gname,
        gname_event_mapping,
        how="left",
        left_on="target",
        right_on="eventid",
    )
    df_target_gname.rename(columns={"gname": "gname_target"}, inplace=True)

    # Filtering and cleaning
    gnames_relations_edgelist = df_target_gname[
        df_target_gname.gname_source != df_target_gname.gname_target
    ]
    gnames_relations_edgelist = gnames_relations_edgelist[
        gnames_relations_edgelist.gname_source != "Unknown"
    ]
    gnames_relations_edgelist = gnames_relations_edgelist[
        gnames_relations_edgelist.gname_target != "Unknown"
    ]
    gnames_relations_edgelist = gnames_relations_edgelist[
        ["gname_source", "gname_target"]
    ]
    gnames_relations_edgelist.dropna(inplace=True)

    G_rel = nx.from_pandas_edgelist(
        gnames_relations_edgelist, source="gname_source", target="gname_target"
    )

    # Merging two graphs
    G = nx.compose(G_country_decade, G_rel)

    return G


def dbscan_hyperparameters(embeddings, e_lower=0.3, e_upper=0.8, m_lower=3, m_upper=15):
    """
    function to run dbscan clustering greedily for different min_samples and e_eps and discover number of resulting clusters and noise points
    """
    n_clusters_res = []
    n_noise_res = []
    e_res = []
    m_res = []

    for e in np.arange(e_lower, e_upper, 0.1):
        print("eps:" + str(e))
        for m in np.arange(m_lower, m_upper, 1):
            db = DBSCAN(eps=e, min_samples=m).fit(embeddings)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            n_clusters_res.append(n_clusters_)
            n_noise_res.append(n_noise_)
            e_res.append(e)
            m_res.append(m)

    plt.plot(n_clusters_res, n_noise_res, "ro")
    plt.xlabel("Number of clusters")
    plt.ylabel("Number of noise points")
    plt.show()

    dt = pd.DataFrame(
        {
            "n_clusters": n_clusters_res,
            "n_noise": n_noise_res,
            "eps": e_res,
            "min_samples": m_res,
        }
    )

    return dt


def cluster_external_internal_edges(G, cluster_df, cluster_name="cluster"):
    nclusters = pd.unique(cluster_df[cluster_name])

    nexternal_edges = []
    ninternal_edges = []
    for cl in nclusters:
        nodes_in_cluster = list(cluster_df.index[cluster_df[cluster_name] == cl].values)
        external = list(nx.edge_boundary(G, nodes_in_cluster))
        nexternal_edges.append(len(external))
        internal = G.subgraph(nodes_in_cluster)
        ninternal_edges.append(len(internal.edges()))
    df = pd.DataFrame(
        {
            "cluster": nclusters,
            "nexternal_edges": nexternal_edges,
            "ninternal_edges": ninternal_edges,
        }
    )
    df["ratio"] = df["ninternal_edges"] / df["nexternal_edges"]
    return df
