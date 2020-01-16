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

__all__ = ["experimental"]

from textwrap import dedent

ISSUE_BASE = "https://github.com/stellargraph/stellargraph/issues"


def render_issue_link(number):
    return f"`#{number} <{ISSUE_BASE}/{number}>`_"


def experimental(*, reason, issues=None):
    """
    A decorator to mark a function, method or class as experimental, meaning it may not be complete.

    Args:
        reason (str): why this is experimental
        issues (list of int, optional): any relevant ``stellargraph/stellargraph`` issues
    """
    if issues is None:
        issues = []

    if issues:
        links = ", ".join(render_issue_link(number) for number in issues)
        issue_text = f" (see: {links})"
    else:
        issue_text = ""

    def decorator(decl):
        # add warning at the start of the documentation
        # <https://docutils.sourceforge.io/docs/ref/rst/directives.html#caution>
        decl.__doc__ = f"""\
.. warning::

   ``{decl.__qualname__}`` is experimental: {reason}{issue_text}. It may be difficult to use and may
   have major changes at any time.

{dedent(decl.__doc__)}
"""
        return decl

    return decorator
