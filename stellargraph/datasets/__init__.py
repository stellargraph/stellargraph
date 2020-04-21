# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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
The datasets package contains classes to download sample datasets

"""

# these are defined explicitly for autodoc to pick them up via :members:
__all__ = [
    "Cora",
    "CiteSeer",
    "PubMedDiabetes",
    "BlogCatalog3",
    "MovieLens",
    "AIFB",
    "MUTAG",
    "PROTEINS",
    "WN18",
    "WN18RR",
    "FB15k",
    "FB15k_237",
    "IAEnronEmployees",
]

from .datasets import *
