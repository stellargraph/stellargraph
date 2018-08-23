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

import pytest
import os
import numpy as np
import networkx as nx
from stellar.data.edge_splitter import EdgeSplitter
from stellar.data.epgm import EPGM
import random
import datetime
from datetime import datetime, timedelta


def create_heterogeneous_graph():
    g = nx.Graph()

    random.seed(42)  # produces the same graph every time

    start_date_dt = datetime.strptime("01/01/2015", "%d/%m/%Y")
    end_date_dt = datetime.strptime("01/01/2017", "%d/%m/%Y")
    start_end_days = (
        end_date_dt - start_date_dt
    ).days  # the number of days between start and end dates

    # 50 nodes of type person
    person_node_ids = list(range(0, 50))
    for person in person_node_ids:
        g.add_node(person, label="user", elite=random.choices([0, 1], k=1)[0])

    # 200 nodes of type paper
    paper_node_ids = list(range(50, 250))
    g.add_nodes_from(paper_node_ids, label="paper")

    # 10 nodes of type venue
    venue_node_ids = list(range(250, 260))
    g.add_nodes_from(venue_node_ids, label="venue")

    # add the person - friend -> person edges
    # each person can be friends with 0 to 5 others; edges include a date
    for person_id in person_node_ids:
        k = random.randrange(5)
        friend_ids = set(random.sample(person_node_ids, k=k)) - {
            person_id
        }  # no self loops
        for friend in friend_ids:
            g.add_edge(
                person_id,
                friend,
                label="friend",
                date=(
                    start_date_dt + timedelta(days=random.randrange(start_end_days))
                ).strftime("%d/%m/%Y"),
            )

    # add the person - writes -> paper edges
    for person_id in person_node_ids:
        k = random.randrange(5)
        paper_ids = random.sample(paper_node_ids, k=k)
        for paper in paper_ids:
            g.add_edge(person_id, paper, label="writes")

    # add the paper - published-at -> venue edges
    for paper_id in paper_node_ids:
        venue_id = random.sample(venue_node_ids, k=1)[
            0
        ]  # paper is published at 1 venue only
        g.add_edge(paper_id, venue_id, label="published-at")

    return g


def read_graph(graph_file, dataset_name, directed=False, weighted=False):
    """
    Reads the input network in networkx.

    :param graph_file: The directory where graph in EPGM format is stored
    :param dataset_name: The name of the graph selected out of all the graph heads in EPGM file
    :return: The graph in networkx format
    """
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

        g = G_epgm.to_nx(graph_id, directed)
        if weighted:
            raise NotImplementedError
        else:
            # This is the correct way to set the edge weight in a MultiGraph.
            edge_weights = {e: 1 for e in g.edges(keys=True)}
            nx.set_edge_attributes(g, name="weight", values=edge_weights)

    except:  # otherwise, assume arg.input points to an edgelist file
        if weighted:
            g = nx.read_edgelist(
                graph_file,
                nodetype=int,
                data=(("weight", float),),
                create_using=nx.DiGraph(),
            )
        else:
            g = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.DiGraph())
            for edge in g.edges():
                g[edge[0]][edge[1]]["weight"] = 1

        if not directed:
            g = g.to_undirected()

    if not nx.is_connected(g):
        print("Graph is not connected")
        # take the largest connected component as the data
        g = max(nx.connected_component_subgraphs(g, copy=True), key=len)
        print(
            "Largest subgraph statistics: {} nodes, {} edges".format(
                g.number_of_nodes(), g.number_of_edges()
            )
        )

    print(
        "Graph statistics: {} nodes, {} edges".format(
            g.number_of_nodes(), g.number_of_edges()
        )
    )
    return g


class TestEdgeSplitterHomogeneous(object):
    print(os.getcwd())
    if os.getcwd().split("/")[-1] == "tests":
        input_dir = os.path.expanduser("resources/data/cora/cora.epgm")
    else:
        input_dir = os.path.expanduser("tests/resources/data/cora/cora.epgm")

    dataset_name = "cora"

    g = read_graph(input_dir, dataset_name)
    g = nx.Graph(g)

    es_obj = EdgeSplitter(g)

    def test_split_data_global(self):
        p = 0.1

        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method="global"
        )

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 1
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

    def test_split_data_local(self):
        p = 0.1
        # using default sampling probabilities
        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method="local"
        )

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

        sampling_probs = [0.0, 0.0, 0.1, 0.2, 0.5, 0.2]
        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method="local", probs=sampling_probs
        )

        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

        sampling_probs = [0.2, 0.1, 0.2, 0.5, 0.2]  # values don't sum to 1
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method="local", probs=sampling_probs)


class TestEdgeSplitterHeterogeneous(object):

    # if os.getcwd().split("/")[-1] == "tests":
    #     input_dir = os.path.expanduser("resources/data/yelp/yelp.epgm")
    # else:
    #     input_dir = os.path.expanduser("tests/resources/data/yelp/yelp.epgm")
    #
    # dataset_name = "small_yelp_example"
    # g = read_graph(input_dir, dataset_name)
    g = create_heterogeneous_graph()

    es_obj = EdgeSplitter(g)

    def test_split_data_by_edge_type_and_attribute(self):
        # test global method for negative edge sampling
        self._test_split_data_by_edge_type_and_attribute(method="global")

        # test local method for positive edge sampling
        self._test_split_data_by_edge_type_and_attribute(method="local")

    def _test_split_data_by_edge_type_and_attribute(self, method):
        p = 0.1
        res = self.es_obj.train_test_split(
            p=p,
            method=method,
            edge_label="friend",
            edge_attribute_label="date",
            attribute_is_datetime=True,
            edge_attribute_threshold="01/01/2008",
        )
        g_test, edge_data_ids_test, edge_data_labels_test = res

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

        with pytest.raises(KeyError):
            # This call will raise an exception because the edges of type friend don't have attribute of type 'Any'
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="Any",
                attribute_is_datetime=True,
                edge_attribute_threshold="01/01/2008",
            )
        with pytest.raises(KeyError):
            # This call will raise and exception because edges of type 'towards' don't have a 'date' attribute
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="published-at",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="01/01/2008",
            )

        with pytest.raises(ValueError):
            # This call will raise an exception because the edge attribute must be specified as datetime
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=False,
                edge_attribute_threshold="01/01/2008",
            )

        # Th below call will raise an exception because the threshold value does not have the correct format dd/mm/yyyy
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="01/2008",
            )
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="Jan 2005",
            )
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="01-01-2000",
            )
        with pytest.raises(ValueError):
            # month is out of range; no such thing as a 14th month in a year
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="01/14/2008",
            )
        with pytest.raises(ValueError):
            # day is out of range; no such thing as a 32nd day in October
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="friend",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="32/10/2008",
            )

        with pytest.raises(Exception):
            # This call to train_test_split will raise an exception because all the edges of type 'writes' are
            # on the minimum spanning tree and cannot be removed.
            self.es_obj.train_test_split(
                p=p,
                method=method,
                edge_label="writes",
                edge_attribute_label="date",
                attribute_is_datetime=True,
                edge_attribute_threshold="01/01/2008",
            )

    def test_split_data_by_edge_type(self):
        # test global method for negative edge sampling
        self._test_split_data_by_edge_type(method="global")

        # test local method for positive edge sampling
        self._test_split_data_by_edge_type(method="local")

    def _test_split_data_by_edge_type(self, method):
        p = 0.1
        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method=method, edge_label="friend"
        )

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

        with pytest.raises(Exception):
            # This call will raise an exception because the graph has no edges of type 'Non Label'
            self.es_obj.train_test_split(p=p, method=method, edge_label="No Label")

        # with pytest.raises(Exception):
        #     # This call to train_test_split will raise an exception because all the edges of type 'writes' are
        #     # on the minimum spanning tree and cannot be removed.
        #     self.es_obj.train_test_split(p=p, method=method, edge_label="writes")

    def test_split_data_global(self):
        p = 0.1
        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method="global"
        )

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)

    def test_split_data_local(self):
        p = 0.1

        # using default sampling probabilities
        g_test, edge_data_ids_test, edge_data_labels_test = self.es_obj.train_test_split(
            p=p, method="local"
        )

        # if all goes well, what are the expected return values?
        num_sampled_positives = np.sum(edge_data_labels_test == 1)
        num_sampled_negatives = np.sum(edge_data_labels_test == 0)

        assert num_sampled_positives > 0
        assert num_sampled_negatives > 0
        assert len(edge_data_ids_test) == len(edge_data_labels_test)
        assert (num_sampled_positives - num_sampled_negatives) <= 2
        assert len(g_test.edges()) < len(self.g.edges())
        assert nx.is_connected(g_test)


class TestEdgeSplitterCommon(object):

    # if os.getcwd().split("/")[-1] == "tests":
    #     input_dir = os.path.expanduser("resources/data/yelp/yelp.epgm")
    # else:
    #     input_dir = os.path.expanduser("tests/resources/data/yelp/yelp.epgm")
    #
    # dataset_name = "small_yelp_example"
    #
    # g = read_graph(input_dir, dataset_name)

    g = create_heterogeneous_graph()

    es_obj = EdgeSplitter(g)

    def test_split_data_p_parameter(self):
        # Test some edge cases for the value of p, e.g., < 0, = 0, > 1, =1
        p = 0
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method="global")

        p = -0.1
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method="global")

        p = 1.001
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method="global")

        p = 1
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method="global")

    def test_split_data_method_parameter(self):
        p = 0.5  # any value in the interval (0, 1) should do
        sampling_method = "other"  # correct values are global and local only
        with pytest.raises(ValueError):
            self.es_obj.train_test_split(p=p, method=sampling_method)
