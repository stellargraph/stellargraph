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
import logging
from shutil import unpack_archive
from urllib.request import urlretrieve
from typing import List, Optional


log = logging.getLogger(__name__)


class DatasetLoader(object):
    """
    Base class for downloading sample datasets.

    This class is used by inherited classes for each specific dataset, providing basic functionality to
    download a dataset from a URL.

    The default download path of ~/data can be changed by setting the STELLARGRAPH_DATASETS_PATH environment variable,
    and each dataset will be downloaded to a subdirectory within this path.
    """

    @classmethod
    def __init_subclass__(
        cls,
        name: str,
        directory_name: str,
        url: str,
        url_archive_format: Optional[str],
        expected_files: List[str],
        description: str,
        source: str,
        data_subdirectory_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        cls.name = name
        cls.directory_name = directory_name
        cls.url = url
        cls.url_archive_format = url_archive_format
        cls.expected_files = expected_files
        cls.description = description
        cls.source = source
        cls.data_subdirectory_name = data_subdirectory_name

        # auto generate documentation
        if cls.__doc__ is not None:
            raise ValueError(
                "DatasetLoader docs are automatically generated and should be empty"
            )
        cls.__doc__ = f"{cls.description}\n\nFurther details at: {cls.source}"

        super().__init_subclass__(**kwargs)

    @property
    def base_directory(self) -> str:
        """str: The path of the directory containing this dataset."""
        return os.path.join(self._all_datasets_directory(), self.directory_name)

    @property
    def data_directory(self) -> str:
        """str: The path of the directory containing the data content files for this dataset."""
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
        """Return the path of the base directory which contains subdirectories for each dataset."""
        return os.path.expanduser(
            os.getenv("STELLARGRAPH_DATASETS_PATH", os.path.join("~", "data"))
        )

    def _resolve_path(self, filename):
        """Convert dataset relative files to their full path on filesystem"""
        return os.path.join(self.base_directory, filename)

    def _is_downloaded(self) -> bool:
        """Returns true if the expected files for the dataset are present"""
        return all(
            os.path.isfile(self._resolve_path(file)) for file in self.expected_files
        )

    def _verify_files_downloaded(self) -> None:
        """
        Raises:
            FileNotFoundError: If any files within dataset are missing.
        """
        missing_files = ",".join(
            [
                file
                for file in self.expected_files
                if not os.path.isfile(self._resolve_path(file))
            ]
        )
        if missing_files:
            raise FileNotFoundError(
                f"{self.name} dataset failed to download file(s): {missing_files}"
            )

    def download(self, ignore_cache: Optional[bool] = False) -> None:
        """
        Download the dataset (if not already downloaded)

        Args:
            ignore_cache bool, optional (default=False): Ignore a cached dataset and force a re-download.

        Raises:
            FileNotFoundError: If the dataset is not successfully downloaded.
        """
        if ignore_cache or not self._is_downloaded():
            log.info(
                "%s dataset downloading to %s from",
                self.name,
                self.base_directory,
                self.url,
            )
            if self.url_archive_format is None:
                # single file to download
                self._create_base_directory()
                destination_filename = os.path.join(
                    self.base_directory, self.expected_files[0]
                )
                urlretrieve(self.url, filename=destination_filename)
            else:
                # archive of files
                filename, _ = urlretrieve(
                    self.url
                )  # this will download to a temporary location
                self._create_base_directory()
                unpack_archive(
                    filename, self._all_datasets_directory(), self.url_archive_format
                )
            self._verify_files_downloaded()
        else:
            log.info("%s dataset is already downloaded", self.name)
