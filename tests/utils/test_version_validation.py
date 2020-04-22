# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

import pytest
from unittest.mock import patch
from stellargraph.utils import validate_notebook_version


def test_current_version_older():
    with pytest.raises(
        ValueError,
        match=".* requires StellarGraph version 987.654.321, but an older version .*/1172",
    ):
        validate_notebook_version("987.654.321")


def test_current_version_newer():
    with pytest.warns(
        DeprecationWarning,
        match=".* is designed for an older StellarGraph version 0.0.0 and may not function correctly with the newer .*/1172",
    ):
        validate_notebook_version("0.0.0")


@patch("stellargraph.utils.version_validation.__version__", new="2.0.0b0")
def test_dev_library_with_release_notebook():
    with pytest.raises(
        ValueError,
        match=".* requires StellarGraph version 2.0.0, but an older version 2.0.0b0.*/1172",
    ):
        validate_notebook_version("2.0.0")
