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

import abc


class Generator(abc.ABC):
    """
    A generator supports creating sequences for input into graph machine learning algorithms via the `flow` method.
    """

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        """
        Create a Keras Sequence or similar input, appropriate for a graph machine learning model.
        """
        ...

    def corruptible_input_indices(self):
        return None

    def corrupt_inputs(self, inputs):
        raise ValueError(f"calling 'corrupt_inputs' on {type(self).__name__} that does not support corrupting inputs")
