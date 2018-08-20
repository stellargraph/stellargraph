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

import argparse


def parse_args():
    """
    Parses the command line arguments.

    Returns:
    """
    parser = argparse.ArgumentParser(
        description="Run link prediction on homogeneous and heterogeneous graphs."
    )

    parser.add_argument(
        "--dataset_name",
        nargs="?",
        default="cora",
        help="The dataset name as stored in graphs.json",
    )

    parser.add_argument(
        "--p",
        nargs="?",
        default=0.1,
        help="Percent of edges to sample for positive and negative examples (valid values 0 < p < 1)",
    )

    parser.add_argument(
        "--subgraph_size",
        nargs="?",
        default=0.1,
        help="Percent of nodes for a subgraph of the input data when --subsample is specified (valid values 0 < subgraph_size < 1)",
    )

    parser.add_argument(
        "--edge_type", nargs="?", default="friend", help="The edge type to predict"
    )

    parser.add_argument(
        "--edge_attribute_label",
        nargs="?",
        default="date",
        help="The attribute label by which to split edges",
    )

    parser.add_argument(
        "--edge_attribute_threshold",
        nargs="?",
        default=None,
        help="Any edge with attribute value less that the threshold cannot be removed from graph",
    )

    parser.add_argument(
        "--attribute_is_datetime",
        dest="attribute_is_datetime",
        action="store_true",
        help="If specified, the edge attribute to split on is considered datetime in format dd/mm/yyyy",
    )

    parser.add_argument(
        "--hin",
        dest="hin",
        action="store_true",
        help="If specified, it indicates that the input graph in a heterogenous network; otherwise, the input graph is assumed homogeneous",
    )

    parser.add_argument(
        "--input_graph",
        nargs="?",
        default="~/Projects/data/cora/cora.epgm/",
        help="Input graph filename",
    )

    parser.add_argument(
        "--output_node_features",
        nargs="?",
        default="~/Projects/data/cora/cora.features/cora.emb",
        help="Input graph filename",
    )

    parser.add_argument(
        "--sampling_method",
        nargs="?",
        default="global",
        help="Negative edge sampling method: local or global",
    )

    parser.add_argument(
        "--sampling_probs",
        nargs="?",
        default="0.0, 0.25, 0.50, 0.25",
        help="Negative edge sample probabilities (for local sampling method) with respect to distance from starting node",
    )

    parser.add_argument(
        "--show_hist",
        dest="show_histograms",
        action="store_true",
        help="If specified, a histogram of the distances between source and target nodes for \
                         negative edge samples will be plotted.",
    )

    parser.add_argument(
        "--subsample",
        dest="subsample_graph",
        action="store_true",
        help="If specified, then the original graph is randomly subsampled to 10% of the original size, \
                        with respect to the number of nodes",
    )

    return parser.parse_args()
