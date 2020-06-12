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

import pytest

import tensorflow as tf
import stellargraph as sg
import numpy as np

ignore_stellargraph_experimental_mark = pytest.mark.filterwarnings(
    r"ignore:StellarGraph\(nodes=..., edges=...\):stellargraph.core.experimental.ExperimentalWarning"
)


def model_save_load(tmpdir, sg_model):
    model = tf.keras.Model(*sg_model.in_out_tensors())

    saving_functions = [
        tf.keras.models.save_model,
        tf.keras.Model.save,
        tf.saved_model.save,
    ]
    loading_functions = [
        tf.keras.models.load_model,
        # tf.saved_model.load doesn't restore the Keras Model object
    ]

    for i, func in enumerate(saving_functions):
        saved_dir = str(tmpdir.join(str(i)))
        func(model, str(saved_dir))

        for func in loading_functions:
            loaded = func(saved_dir, sg.custom_keras_layers)

            orig_weights = model.get_weights()
            new_weights = loaded.get_weights()
            assert len(orig_weights) == len(new_weights)
            for orig, new in zip(orig_weights, new_weights):
                np.testing.assert_array_equal(orig, new)
