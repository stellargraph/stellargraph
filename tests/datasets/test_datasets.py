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
import tempfile
import os
from stellargraph.datasets import Cora, CiteSeer
from urllib.error import URLError
from stellargraph.datasets.dataset_loader import DatasetLoader
from urllib.request import urlretrieve
from unittest.mock import patch


# use parametrize to automatically test each of the datasets that (directly) derive from DatasetLoader
@pytest.mark.parametrize("dataset_class", list(DatasetLoader.__subclasses__()))
def test_dataset_download(dataset_class):
    dataset_class().download(ignore_cache=True)


@patch(
    "stellargraph.datasets.datasets.Cora.url", new="http://stellargraph-invalid-url/x"
)
def test_invalid_url() -> None:
    with pytest.raises(URLError):
        Cora().download(ignore_cache=True)


# we add an additional expected file that should break the download
@patch(
    "stellargraph.datasets.datasets.Cora.expected_files",
    new=Cora.expected_files + ["test-missing-file.xyz"],
)
def test_missing_files() -> None:
    # download - the url should work, but the files extracted won't be correct
    with pytest.raises(FileNotFoundError):
        Cora().download()


def test_environment_path_override(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as new_datasets_path:
        monkeypatch.setenv("STELLARGRAPH_DATASETS_PATH", new_datasets_path)
        dataset = CiteSeer()
        assert dataset.base_directory == os.path.join(
            new_datasets_path, dataset.directory_name
        )
        dataset.download()


@patch("stellargraph.datasets.dataset_loader.urlretrieve", wraps=urlretrieve)
def test_download_cache(mock_urlretrieve) -> None:
    # forcing a re-download should call urlretrieve
    Cora().download(ignore_cache=True)
    assert mock_urlretrieve.called

    mock_urlretrieve.reset_mock()

    # if already downloaded and in the cache, then another download should skip urlretrieve
    Cora().download()
    assert not mock_urlretrieve.called
