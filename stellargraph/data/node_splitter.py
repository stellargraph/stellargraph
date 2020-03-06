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

__all__ = ["train_val_test_split", "NodeSplitter"]


_MSG = "this functionality has been removed; please use 'sklearn' or 'pandas', such as 'sklearn.model_selection.train_test_split'"


# Easier functional interface for the splitter:
def train_val_test_split(
    G,
    node_type=None,
    test_size=0.4,
    train_size=0.2,
    targets=None,
    split_equally=False,
    seed=None,
):
    raise ValueError(_MSG)


class NodeSplitter(object):
    def __init__(self):
        raise ValueError(_MSG)
