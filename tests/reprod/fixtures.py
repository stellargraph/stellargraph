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


def assert_reproducible(func, equals, num_iter=20):
    """
    Assert results from calling ``func`` are reproducible. The ``equals`` function is used to test equality of results.

    Args:
        func (callable): Function to check for reproducible "result"
        equals (callable): Function to take two arguments (of "result" type) and check that they are equal
        num_iter (int, default 20): Number of iterations to run through to validate reproducibility.

    """
    out = func()
    for i in range(num_iter):
        out_new = func()
        if not equals(out, out_new):
            assert False, (out, out_new)
    assert True
