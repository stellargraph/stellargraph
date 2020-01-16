# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

"""
Enable easy loading of sample datasets for demonstrations
"""

import os
from shutil import unpack_archive
from urllib.request import urlretrieve
from typing import List, Optional


class DatasetLoader(object):
    """ A class to download sample datasets"""

    def __init__(
        self,
        name: str,
        directory_name: str,
        url: str,
        url_archive_format: Optional[str],
        expected_files: List[str],
        description: str,
        source: str,
        data_subdirectory_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.directory_name = directory_name
        self.url = url
        self.url_archive_format = url_archive_format
        self.expected_files = expected_files
        self.description = description
        self.source = source
        self.data_subdirectory_name = data_subdirectory_name

    @property
    def base_directory(self) -> str:
        """Return the path of the data directory for this dataset"""
        return os.path.join(self._all_datasets_directory(), self.directory_name)

    @property
    def data_directory(self) -> str:
        """Return the directory containing the data content files"""
        if self.data_subdirectory_name is None:
            return self.base_directory
        else:
            return os.path.join(self.base_directory, self.data_subdirectory_name)

    def _create_base_directory(self) -> None:
        data_dir = self.base_directory
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    @staticmethod
    def _all_datasets_directory() -> str:
        """Return the path of the base directory which contains subdirectories for each dataset"""
        return os.path.expanduser(
            os.getenv("STELLARGRAPH_DATASETS_PATH", os.path.join("~", "data"))
        )

    def _is_downloaded(self) -> bool:
        """Returns true if the expected files for the dataset are present"""
        for file in self.expected_files:
            if not os.path.isfile(os.path.join(self.base_directory, file)):
                return False
        return True

    def download(self, ignore_cache: bool = False) -> None:
        """Download the dataset (if not already downloaded, unless ignore_cache=True)"""
        if not self._is_downloaded() or ignore_cache:
            print(
                f"{self.name} dataset downloading to {self.base_directory} from {self.url}"
            )
            if self.url_archive_format is None:
                # single file to download
                assert len(self.expected_files) == 1
                self._create_base_directory()
                destination_filename = os.path.join(self.base_directory, self.expected_files[0])
                urlretrieve(self.url, filename=destination_filename)
            else:
                # archive of files
                filename, _ = urlretrieve(self.url)
                self._create_base_directory()
                unpack_archive(filename, self._all_datasets_directory(), self.url_archive_format)
            if not self._is_downloaded():
                print(f"{self.name} dataset failed to download")
        else:
            print(f"{self.name} dataset is already downloaded")
