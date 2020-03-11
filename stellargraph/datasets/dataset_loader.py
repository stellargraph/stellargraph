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
from shutil import unpack_archive, move
from urllib.request import urlretrieve
from typing import List, Optional, Any
from urllib.error import URLError


log = logging.getLogger(__name__)


class DatasetLoader:
    """
    Base class for downloading sample datasets.

    This class is used by inherited classes for each specific dataset, providing basic functionality to
    download a dataset from a URL.

    The default download path of ~/stellargraph-datasets can be changed by setting the STELLARGRAPH_DATASETS_PATH environment variable,
    and each dataset will be downloaded to a subdirectory within this path.
    """

    # define these for mypy benefit (will be set for derived classes in __init_subclass__ below)
    name = ""
    directory_name = ""
    url = ""
    url_archive_format: Optional[str] = None
    url_archive_contains_directory: bool = True
    expected_files: List[str] = []
    description = ""
    source = ""
    data_subdirectory_name: Optional[str] = None

    @classmethod
    def __init_subclass__(
        cls,
        *,
        name: str,
        directory_name: str,
        url: str,
        url_archive_format: Optional[str],
        url_archive_contains_directory: bool = True,
        expected_files: List[str],
        description: str,
        source: str,
        data_subdirectory_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Used to set class variables during the class definition of derived classes and generate customised docs.
        NOTE: this is not compatible with python's ABC abstract base class, so this class derives from object."""
        cls.name = name
        cls.directory_name = directory_name
        cls.url = url
        cls.url_archive_format = url_archive_format
        cls.url_archive_contains_directory = url_archive_contains_directory
        cls.expected_files = expected_files
        cls.description = description
        cls.source = source
        cls.data_subdirectory_name = data_subdirectory_name

        if url_archive_format is None and len(expected_files) != 1:
            raise ValueError(
                "url_archive_format is None, which requires a single expected_file, found: {expected_files!r}"
            )

        # auto generate documentation
        if cls.__doc__ is not None:
            raise ValueError(
                "DatasetLoader docs are automatically generated and should be empty"
            )
        cls.__doc__ = f"{cls.description}\n\nFurther details at: {cls.source}"

        super().__init_subclass__(**kwargs)  # type: ignore

    def __init__(self) -> None:
        # basic check since this is effectively an abstract base class, and derived classes should have set name
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} can't be instantiated directly, please use a derived class"
            )

    @property
    def base_directory(self) -> str:
        """str: The full path of the directory containing this dataset."""
        return os.path.join(self._all_datasets_directory(), self.directory_name)

    @property
    def data_directory(self) -> str:
        """str: The full path of the directory containing the data content files for this dataset."""
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
        return os.getenv(
            "STELLARGRAPH_DATASETS_PATH",
            os.path.expanduser(os.path.join("~", "stellargraph-datasets")),
        )

    def _resolve_path(self, filename: str) -> str:
        """Convert dataset relative file names to their full path on filesystem"""
        return os.path.join(self.base_directory, filename)

    def _resolve_unpack_path(self):
        if self.url_archive_contains_directory:
            return self._all_datasets_directory()
        else:
            return self.base_directory

    def _missing_files(self) -> List[str]:
        """Returns a list of files that are missing"""
        return [
            file
            for file in self.expected_files
            if not os.path.isfile(self._resolve_path(file))
        ]

    def _is_downloaded(self) -> bool:
        """Returns true if the expected files for the dataset are present"""
        return len(self._missing_files()) == 0

    def _delete_existing_files(self) -> None:
        """ Delete the files for this dataset if they already exist """
        for file in self.expected_files:
            try:
                os.remove(self._resolve_path(file))
            except OSError:
                pass

    def download(self, ignore_cache: Optional[bool] = False) -> None:
        """
        Download the dataset (if not already downloaded)

        Args:
            ignore_cache (bool, optional): Ignore a cached dataset and force a re-download.

        Raises:
            FileNotFoundError: If the dataset is not successfully downloaded.
        """
        if ignore_cache:
            self._delete_existing_files()  # remove any existing dataset files to ensure we re-download

        if ignore_cache or not self._is_downloaded():
            log.info(
                "%s dataset downloading to %s from %s",
                self.name,
                self.base_directory,
                self.url,
            )
            temporary_filename, _ = urlretrieve(self.url)
            if self.url_archive_format is None:
                # not an archive, so the downloaded file is our data and just needs to be put into place
                self._create_base_directory()
                move(temporary_filename, self._resolve_path(self.expected_files[0]))
            else:
                # an archive to unpack.  The folder is created by unpack_archive - therefore the
                # directory_name for this dataset must match the directory name inside the archive file
                unpack_archive(
                    temporary_filename,
                    self._resolve_unpack_path(),
                    self.url_archive_format,
                )
            # verify the download
            missing_files = self._missing_files()
            if missing_files:
                missing = ", ".join(missing_files)
                raise FileNotFoundError(
                    f"{self.name} dataset failed to download file(s): {missing} to {self.data_directory}"
                )
        else:
            log.info("%s dataset is already downloaded", self.name)
