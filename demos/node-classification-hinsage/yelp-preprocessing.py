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
This script preprocesses and filters the Yelp dataset to convert the attributes
of users and businesses to numeric features for use with graph machine-learning.

This script requires the Yelp dataset as input which can be obtained by from the
Yelp Dataset website: https://www.yelp.com/dataset
To download the dataset, select "Download Dataset" on the Yelp Dataset website,
sign the licence agreement, and download the JSON dataset from the download page.
FInally, uncompress the file to an appropriate location.

To use this script, supply the location that you uncompressed the dataset,
(this location should contain "yelp_academic_dataset_user.json")
the output directory (-o) and any other optional arguments:

Example usage:
    python yelp_preprocessing -l <path_to_yelp_dataset> -o .

By default the script will filter the graph to contain only businesses in the state
of Wisconsin. To change this to another state, or to "false" to use the entire dataset
(warning: this will make the machine learning example run very slowly).

Example usage to run without filtering:
    python yelp_preprocessing -l <path_to_yelp_dataset> -o . --filter_state=false

"""
import os
import json
import argparse
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from keras.utils.generic_utils import Progbar
from sklearn import preprocessing, feature_extraction, pipeline

user_feature_names = [
    "review_count",
    "useful",
    "funny",
    "cool",
    "fans",
    "average_stars",
    "compliment_hot",
    "compliment_more",
    "compliment_profile",
    "compliment_cute",
    "compliment_list",
    "compliment_note",
    "compliment_plain",
    "compliment_cool",
    "compliment_funny",
    "compliment_writer",
    "compliment_photos",
]

user_target_name = "elite"

business_feature_names = ["review_count", "stars"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Yelp data processing for Yelp HinSAGE demo"
    )
    parser.add_argument(
        "-l",
        "--location",
        nargs="?",
        type=str,
        default=None,
        help="Location of Yelp JSON data files",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        type=str,
        default=None,
        help="Location to store the output, this should be an existing directory",
    )
    parser.add_argument(
        "-s",
        "--filter_state",
        nargs="?",
        type=str,
        default="WI",
        help="Filter yelp dataset to only contain records connected to this US State",
    )
    parser.add_argument("--users_only", action="store_true", help="Create a graph with only users")
    parser.add_argument("--save_preprocessors", action="store_true", help="Save the fitted data preprocessors")
    parser.add_argument("--use_sparse", action="store_true", help="Save features as sparse matrices")
    args = parser.parse_args()

    # Parameters
    yelp_location = args.location
    output_location = args.output
    filter_state = args.filter_state.lower()
    filter_friends = True
    use_sparse = args.use_sparse

    # Check mandatory arguments
    if not yelp_location:
        raise RuntimeError("Please specify the location of the Yelp dataset with the -l argument")
    if not output_location:
        raise RuntimeError("Please specify the output directory with the -o argument")

    # Check directories:
    if not os.path.isdir(output_location):
        raise OSError(
            "The specified output location doesn't exist or isn't a directory"
        )

    # Load user data from json to a dictionary with the user_id as key
    print("Loading user data")
    with open(os.path.join(yelp_location, "yelp_academic_dataset_user.json"), "r") as f:
        user_data_raw = {}
        for line in f:
            d = json.loads(line)
            user_data_raw[d["user_id"]] = d

    # Load business data from json to a dictionary with the business_id as key
    print("Loading business data")
    with open(
        os.path.join(yelp_location, "yelp_academic_dataset_business.json"), "r"
    ) as f:
        business_data_raw = {}
        for line in f:
            d = json.loads(line)
            business_data_raw[d["business_id"]] = d

    # Load review data from json to a dictionary with the review_id as key
    print("Loading review data")
    with open(
        os.path.join(yelp_location, "yelp_academic_dataset_review.json"), "r"
    ) as f:
        review_data_raw = {}
        for line in f:
            d = json.loads(line)
            review_data_raw[d["review_id"]] = d

    # Create NX graph
    G = nx.Graph()

    # Maintain a list of users, businesses and reviews in graphs
    # these are sets to make inclusion checking fast
    users_in_graph = set()
    business_in_graph = set()
    reviews_in_graph = set()

    # Create business nodes
    print("\nAdding businesses to graph")
    p = Progbar(len(business_data_raw))
    for ii, b in enumerate(business_data_raw.values()):
        p.update(ii)
        if filter_state != "false" and b["state"].lower() != filter_state:
            continue
        G.add_node(b["business_id"], ntype="business")

        # Add business to set
        business_in_graph.add(b["business_id"])

    # Create review links
    print("\nAdding reviews to graph")
    p = Progbar(len(review_data_raw))
    for ii, r in enumerate(review_data_raw.values()):
        p.update(ii)
        if r["business_id"] not in business_in_graph:
            continue

        # Add the user node
        G.add_node(r["user_id"], ntype="user")

        # Add the review edge
        G.add_edge(r["user_id"], r["business_id"], etype="review")

        # Add user and review to sets
        users_in_graph.add(r["user_id"])
        reviews_in_graph.add(r["review_id"])

        # Create friendship graph
    print("\nAdding users to graph")
    p = Progbar(len(user_data_raw))
    for ii, u in enumerate(user_data_raw.values()):
        p.update(ii)
        if u["user_id"] not in users_in_graph:
            continue

        # Add node as type user
        G.add_node(u["user_id"], ntype="user")

        # Connect to friends
        if u["friends"] != "None":
            friend_list = u["friends"].split(", ")

            # optionally include friend nodes not connected to reviews
            if filter_friends:
                friend_list = [f for f in friend_list if f in users_in_graph]
            else:
                users_in_graph.update(friend_list)

            # Add friend nodes and friend edges to graph
            G.add_nodes_from(friend_list, ntype="user")
            G.add_edges_from([(u["user_id"], f) for f in friend_list], etype="friend")

    # --- Convert User Features ---
    print("\nConverting user features")

    # Extract user data and user_ids
    user_ids = list(users_in_graph)
    user_attributes = [
        {k: v for k, v in user_data_raw[uid].items() if k in user_feature_names}
        for uid in user_ids
    ]

    # Preprocess user features using Scikit-Learn
    # Note we use a nonlinear transform as the user features as mostly counts
    # which are highly non-normal.
    uf_extract = feature_extraction.DictVectorizer(sparse=use_sparse)
    uf_transform = preprocessing.FunctionTransformer(np.log1p, np.expm1)
    uf_encoder = pipeline.Pipeline([("extract", uf_extract), ("scale", uf_transform)])
    user_features = uf_encoder.fit_transform(user_attributes)

    # Create a Pandas dataframe to store features
    user_features = pd.DataFrame(user_features, index=user_ids)
    del user_attributes

    # Get user targets:

    # 'elite' attribute is a comma separated list of years that they are elite
    # target_data = [
    #     {k: 1 for k in user_data_raw[uid][user_target_name].split(", ")}
    #     for uid in user_ids
    # ]

    # Transform elite attribute to simple true and false:
    # note that the original data contains a list of years they have elite status
    target_data = [
        {
            user_target_name: "false"
            if user_data_raw[uid][user_target_name] == "None"
            else "true"
        }
        for uid in user_ids
    ]
    target_encoder = feature_extraction.DictVectorizer(sparse=use_sparse)
    user_targets = target_encoder.fit_transform(target_data)
    del target_data
    del user_data_raw

    # Store as a Pandas dataframe
    user_targets = pd.DataFrame(user_targets, index=user_ids)

    # --- Convert Business Features ---
    print("Converting business features")

    # Extract features for business:
    business_ids = list(business_in_graph)
    business_attributes = [
        {k: v for k, v in business_data_raw[bid].items() if k in business_feature_names}
        for bid in business_ids
    ]

    # Preprocess business features using Scikit-Learn
    # Note we use a nonlinear transform as the user features as mostly counts
    # which are highly non-normal.
    bf_encoding = feature_extraction.DictVectorizer(sparse=use_sparse)
    bf_transform = preprocessing.FunctionTransformer(np.log1p, np.expm1)
    # bf_transform = preprocessing.QuantileTransformer(n_quantiles=10)
    bf_encoder = pipeline.Pipeline([("extract", bf_encoding), ("scale", bf_transform)])
    business_attribute_features = bf_encoder.fit_transform(business_attributes).astype(
        "float32"
    )

    # Extract features for business categories separately.
    business_categories = [
        {k: 1 for k in business_data_raw[bid]["categories"].split(", ")}
        if business_data_raw[bid]["categories"] is not None
        else {}
        for bid in business_ids
    ]
    bc_encoder = feature_extraction.DictVectorizer(sparse=False)
    business_category_features = bc_encoder.fit_transform(business_categories).astype(
        "float32"
    )

    # Concatenate business features and business categories
    business_features = np.concatenate(
        [business_attribute_features, business_category_features], axis=1
    )
    del business_data_raw
    del business_attributes
    del business_attribute_features
    del business_category_features

    business_features = pd.DataFrame(business_features, index=business_ids)

    print(
        "Number of users: {}, number of businesses: {}, number of reviews: {}".format(
            len(users_in_graph), len(business_in_graph), len(reviews_in_graph)
        )
    )

    # Optional: Save the encoders to apply to different data
    encodings = {
        "user_feature": uf_encoder,
        "user_target": target_encoder,
        "business_feature": bf_encoder,
        "business_category": bc_encoder,
    }
    with open(os.path.join(output_location, "yelp_preprocessing.pkl"), "wb") as f:
        pickle.dump(encodings, f)

    # Save features
    print("\nSaving feature data")
    user_features.to_pickle(os.path.join(output_location, "user_features_filtered.pkl"))
    business_features.to_pickle(
        os.path.join(output_location, "business_features_filtered.pkl")
    )
    user_targets.to_pickle(os.path.join(output_location, "user_targets_filtered.pkl"))

    # Save graph
    graph_location = os.path.join(output_location, "yelp_graph_filtered.graphml")
    print("Saving graph as {}".format(graph_location))
    nx.write_graphml(G, graph_location)
