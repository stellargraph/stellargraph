# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
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

import stellargraph as sg
import inspect
import tensorflow as tf


def _defined_in_stellargraph(obj):
    try:
        print(obj, inspect.getfile(obj), inspect.getmodule(obj))
        return "stellargraph" in inspect.getmodule(obj).__name__
    except TypeError:
        return False


def _find_stellargraph_classes(obj, visited=None):
    if visited is None:
        visited = set()

    if id(obj) in visited:
        return

    visited.add(id(obj))

    if inspect.isclass(obj):
        yield obj
        return

    def pred(sub_obj):
        class_or_module = inspect.ismodule(sub_obj) or inspect.isclass(sub_obj)
        return class_or_module and _defined_in_stellargraph(sub_obj)

    for _name, sub_obj in inspect.getmembers(obj, pred):
        yield from _find_stellargraph_classes(sub_obj, visited)


def test_custom_keras_layers_includes_all_layers():
    all_stellargraph_classes = _find_stellargraph_classes(sg)

    all_layers = {
        layer
        for layer in all_stellargraph_classes
        if issubclass(layer, tf.keras.layers.Layer)
    }

    assert set(sg.custom_keras_layers.values()) == all_layers
