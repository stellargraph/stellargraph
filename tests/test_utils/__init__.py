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

from tensorflow import keras
import stellargraph as sg
import numpy as np

ignore_stellargraph_experimental_mark = pytest.mark.filterwarnings(
    r"ignore:StellarGraph\(nodes=..., edges=...\):stellargraph.core.experimental.ExperimentalWarning"
)


def model_save_load(tmpdir, sg_model):
    model = keras.Model(*sg_model.in_out_tensors())

    model.summary()
    save_model_dir = tmpdir.join("save_model")
    keras.models.save_model(model, str(save_model_dir))

    save_dir = tmpdir.join("save")
    model.save(str(save_dir))

    for saved_dir in [save_model_dir, save_dir]:
        loaded = keras.models.load_model(str(saved_dir), sg.custom_keras_layers)

        for orig, new in zip(model.get_weights(), loaded.get_weights()):
            np.testing.assert_array_equal(orig, new)
