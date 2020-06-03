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

# this test has to run first, because of the `tf.debugging.set_log_device_placement` call, hence the
# file name is alphabetically early.

import os
import tensorflow as tf
import pytest
from . import require_gpu

# When the environment variable is set, we need to be sure that we're running on a GPU
@pytest.mark.skipif(
    not require_gpu,
    reason="STELLARGRAPH_MUST_USE_GPU is not set to 1, so a GPU does not have to be used",
)
def test_on_gpu_when_requested():
    # we can't easily capture this
    tf.debugging.set_log_device_placement(True)

    # check we can execute something on it
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    c = tf.matmul(a, b)

    assert c.numpy().shape == (2, 2)

    assert tf.config.list_physical_devices("GPU")
