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

import networkx as nx


class StellarGraph(nx.MultiGraph):
    """
    Our own class for heterogeneous undirected graphs, inherited from nx.MultiGraph, with extra stuff to be added that's needed by samplers and mappers
    """

    def __init__(self):
        super().__init__()


class StellarDiGraph(nx.MultiDiGraph):
    """
    Our own class for heterogeneous directed graphs, inherited from nx.MultiDiGraph, with extra stuff to be added that's needed by samplers and mappers
    """

    def __init__(self):
        super().__init__()


