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
import re
from .. import __version__


__all__ = ["validate_notebook_version"]

# regex version parsing to avoid dependency of packaging module
_VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>[-a-z].*)?$"
)


class _Suffix:
    """The suffix of a version, to be able to control the sort order"""

    def __init__(self, suffix):
        self.suffix = suffix

    def __eq__(self, other):
        return self.suffix == other.suffix

    def __lt__(self, other):
        # any suffix is less than no suffix (1.2.3b0 < 1.2.3)
        if self.suffix is None:
            return False
        if other.suffix is None:
            return True
        return self.suffix < other.suffix


def _parse(version):
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"string: expected a valid version, found {version!r}")

    major = int(match["major"])
    minor = int(match["minor"])
    patch = int(match["patch"])
    suffix = _Suffix(match["suffix"])
    return major, minor, patch, suffix


def _display_in_notebook(message):
    try:
        import IPython
    except ImportError:
        # IPython's not installed, so mustn't be running in a jupyter context
        return
    IPython.display.display(IPython.display.Markdown(message))


def validate_notebook_version(notebook_version):
    """
    Validate a notebook created for a specific version of StellarGraph.

    Args:
        notebook_version(str): the library version that the notebook was created for
    """
    version_stellargraph = _parse(__version__)
    version_notebook = _parse(notebook_version)
    help_url = "https://github.com/stellargraph/stellargraph/issues/1172"

    def _format_for_error(m):
        return f"{m} Please see: <{help_url}>."

    def _format_for_html(m):
        return f'<div class="alert alert-block alert-danger">{m} Please see: <a href="{help_url}">{help_url}</a>.</div>'

    if version_stellargraph < version_notebook:
        message = f"This notebook requires StellarGraph version {notebook_version}, but an older version {__version__} is installed."
        _display_in_notebook(_format_for_html(message))
        raise ValueError(_format_for_error(message)) from None
    elif version_stellargraph > version_notebook:
        message = f"This notebook is designed for an older StellarGraph version {notebook_version} and may not function correctly with the newer installed version {__version__}."
        _display_in_notebook(_format_for_html(message))
        warnings.warn(
            _format_for_error(message), DeprecationWarning, stacklevel=2,
        )
