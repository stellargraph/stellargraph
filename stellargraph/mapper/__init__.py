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
The mapper package contains classes and functions to map graph data to neural network inputs

"""

# __all__ = ["link_mappers", "node_mappers"]

# Expose the generators
from .base import Generator
from .sequences import *
from .sampled_link_generators import *
from .sampled_node_generators import *
from .full_batch_generators import *
from .mini_batch_node_generators import *
from .graphwave_generator import *
from .adjacency_generators import *
from .knowledge_graph import *
from .padded_graph_generator import *
from .corrupted import *
from .sliding import *
