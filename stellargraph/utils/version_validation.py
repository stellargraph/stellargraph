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

import warnings
from packaging import version
from ..version import __version__

__all__ = ["validate_notebook_version"]


def validate_notebook_version(notebook_version):
    """
    Validate a notebook created for a specific version of StellarGraph.

    Args:
        notebook_version(str): the library version that the notebook was created for
    """
    version_stellargraph = version.parse(__version__)
    version_notebook = version.parse(notebook_version)
    extra_info = (
        "Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    )
    if version_stellargraph < version_notebook:
        raise ValueError(
            f"This notebook requires StellarGraph version {notebook_version}, but an older version {__version__} is installed. {extra_info}"
        ) from None
    elif version_stellargraph > version_notebook:
        warnings.warn(
            f"This notebook is designed for an older StellarGraph version {notebook_version} and may not function correctly with the newer installed version {__version__}. {extra_info}",
            DeprecationWarning,
            stacklevel=2,
        )
