# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
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

from textwrap import dedent


def experimental(*, reason):
    def decorator(decl):
        # add warning at the start of the documentation
        # <https://docutils.sourceforge.io/docs/ref/rst/directives.html#caution>
        decl.__doc__ = f"""\
.. warning::

   ``{decl.__qualname__}`` is experimental: {reason}. It may be difficult to use and may have major
   changes at any time.

{dedent(decl.__doc__)}
"""
        return decl

    return decorator
