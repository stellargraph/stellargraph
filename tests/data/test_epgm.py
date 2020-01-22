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

# Tests of EPGM class defined in epgm.py
# Author: Yuriy Tyshetskiy

import pytest
import os
import numpy as np
from stellargraph.data.epgm import EPGM


class Test_EPGM_IO_Homogeneous(object):
    """Test IO operations on homogeneous EPGM graphs"""

    if os.getcwd().split("/")[-1] == "tests":
        input_dir = os.path.expanduser("./resources/data/cora/cora.epgm")
    else:
        input_dir = os.path.expanduser("./tests/resources/data/cora/cora.epgm")

    dataset_name = "cora"
    node_type = "paper"
    target_attribute = "subject"
    epgm_input = True

    def test_load_epgm(self):
        """Test that the EPGM is loaded correctly from epgm path"""
        G_epgm = EPGM(self.input_dir)
        print(self.input_dir)

        assert "graphs" in G_epgm.G.keys()
        assert "vertices" in G_epgm.G.keys()
        assert "edges" in G_epgm.G.keys()

        # check that G_epgm.G['graphs] has at least one graph head:
        assert len(G_epgm.G["graphs"]) > 0

        # cora nodes should have a subject attribute
        graph_id = G_epgm.G["graphs"][0]["id"]
        assert self.target_attribute in G_epgm.node_attributes(graph_id, self.node_type)

        # cora should have 2708 vertices
        n_nodes = 2708
        nodes = G_epgm.G["vertices"]
        assert len(nodes) == n_nodes

        # cora nodes should have 7 unique values for subject attribute:
        assert sum(["data" in v for v in nodes]) == n_nodes
        subjects = np.unique([v["data"][self.target_attribute] for v in nodes])
        assert len(subjects) == 7

    def test_node_types(self):
        """Test the .node_types() method"""
        G_epgm = EPGM(self.input_dir)
        graph_id = G_epgm.G["graphs"][0]["id"]

        # cora has a single 'paper' node type:
        node_types = G_epgm.node_types(graph_id)

        assert len(node_types) == 1
        assert self.node_type in node_types

        with pytest.raises(Exception):
            G_epgm.node_types("invalid_graph_id")

    def test_node_attributes(self):
        """Test the .node_attributes() method"""
        G_epgm = EPGM(self.input_dir)
        graph_id = G_epgm.G["graphs"][0]["id"]

        # cora has 1433 unique node attributes, including 'subject'
        node_attributes = G_epgm.node_attributes(graph_id, self.node_type)

        assert self.target_attribute in node_attributes

        # after the predictions cora has 1434 attributes, including subject and subject_PREDICTED
        if self.epgm_input:
            assert (
                len(node_attributes) == 1433
            ), "There should be 1433 unique node attributes; found {}".format(
                len(node_attributes)
            )
        else:
            assert (
                len(node_attributes) == 1434
            ), "There should be 1434 unique node attributes; found {}".format(
                len(node_attributes)
            )

        # passing a non-existent node type should return an empty array of node attributes:
        assert len(G_epgm.node_attributes(graph_id, "person")) == 0

        # if node_type is not supplied, a TypeError should be raised:
        with pytest.raises(TypeError):
            G_epgm.node_attributes(graph_id)


class Test_EPGM_IO_Heterogeneous(object):
    """Test IO operations on heterogeneous EPGM graphs"""

    if os.getcwd().split("/")[-1] == "tests":
        input_dir = os.path.expanduser("./resources/data/hin_random/")
    else:
        input_dir = os.path.expanduser("./tests/resources/data/hin_random")

    dataset_name = "hin"
    node_type = "person"
    target_attribute = "elite"

    def test_load_epgm(self):
        """Test that the EPGM is loaded correctly from epgm path"""
        G_epgm = EPGM(self.input_dir)

        assert "graphs" in G_epgm.G.keys()
        assert "vertices" in G_epgm.G.keys()
        assert "edges" in G_epgm.G.keys()

        # check that G_epgm.G['graphs] has at least one graph head:
        assert len(G_epgm.G["graphs"]) > 0

        # graph nodes of self.node_type type should have a self.target_attribute attribute
        graph_id = G_epgm.G["graphs"][0]["id"]
        assert self.target_attribute in G_epgm.node_attributes(graph_id, self.node_type)

        # graph should have 260 vertices
        n_nodes = 260
        nodes = G_epgm.G["vertices"]
        assert len(nodes) == n_nodes

        # 'user' nodes should have 3 unique values for 'elite' attribute:
        # first make sure that all nodes have 'data' key
        assert sum(["data" in v for v in nodes]) == n_nodes
        labels_all = [v["data"].get(self.target_attribute) for v in nodes]
        labels = list(filter(lambda l: l is not None, labels_all))
        assert len(np.unique(labels)) == 3

    def test_node_types(self):
        """Test the .node_types() method"""
        G_epgm = EPGM(self.input_dir)
        graph_id = G_epgm.G["graphs"][0]["id"]

        # dataset has multiple node types:
        node_types = G_epgm.node_types(graph_id)

        assert len(node_types) == 3
        assert "person" in node_types
        assert "paper" in node_types
        assert "venue" in node_types

        with pytest.raises(Exception):
            G_epgm.node_types("invalid_graph_id")

    def test_node_attributes(self):
        """Test the .node_attributes() method"""
        G_epgm = EPGM(self.input_dir)
        graph_id = G_epgm.G["graphs"][0]["id"]

        # dataset has 1 unique 'user' node attribute, 'elite'
        node_attributes = G_epgm.node_attributes(graph_id, self.node_type)

        assert self.target_attribute in node_attributes
        assert (
            len(node_attributes) == 1
        ), "There should be 1 unique node attribute; found {}".format(
            len(node_attributes)
        )

        # passing a non-existent node type should return an empty array of node attributes:
        assert len(G_epgm.node_attributes(graph_id, "business")) == 0

        # if node_type is not supplied, a TypeError should be raised:
        with pytest.raises(TypeError):
            G_epgm.node_attributes(graph_id)


class Test_EPGMOutput(Test_EPGM_IO_Homogeneous):
    """Tests for the epgm produced by epgm_writer"""

    if os.getcwd().split("/")[-1] == "tests":
        input_dir = os.path.expanduser("./resources/data/cora/cora.out")
    else:
        input_dir = os.path.expanduser("./tests/resources/data/cora/cora.out")

    epgm_input = False
