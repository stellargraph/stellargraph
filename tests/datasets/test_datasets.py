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
import shutil
import os
from stellargraph.datasets import Cora, CiteSeer
from stellargraph.datasets.dataset_loader import download_all_datasets
from urllib.error import URLError


def test_re_download_all_datasets() -> None:
    # to force re-downloading, we ignore the cached datasets
    # note that this is fairly slow, as it will re-download all of our demo datasets
    assert download_all_datasets(ignore_cache=True)


def test_invalid_url() -> None:
    dataset = Cora()
    dataset.url = "http://stellargraph-invalid-url/x"
    with pytest.raises(URLError):
        dataset.download(ignore_cache=True)


def test_missing_files() -> None:
    # we start with Cora to a special path, but with the URL from CiteSeer
    dataset = Cora()
    dataset.directory_name = "test-missing-files"
    dataset.url = CiteSeer().url

    # make sure the target directory is empty
    def remove_dataset_directory():
        if os.path.exists(dataset.base_directory):
            shutil.rmtree(dataset.base_directory)

    remove_dataset_directory()
    # download - the url should work, but the files extracted won't be correct
    with pytest.raises(FileNotFoundError):
        dataset.download()
    remove_dataset_directory()  # cleanup
