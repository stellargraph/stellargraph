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
import functools

__all__ = ["experimental", "ExperimentalWarning"]

from textwrap import dedent
import warnings

ISSUE_BASE = "https://github.com/stellargraph/stellargraph/issues"


class ExperimentalWarning(Warning):
    pass


def render_link(number, for_rst):
    link = f"{ISSUE_BASE}/{number}"
    if for_rst:
        return f"`#{number} <{link}>`_"
    return link


def issue_text(issues, for_rst):
    if issues:
        links = ", ".join(render_link(number, for_rst) for number in issues)
        return f" (see: {links})"
    else:
        return ""


def messages(decl, reason, issues):
    def description(for_rst):
        return (
            f"is experimental: {reason}{issue_text(issues, for_rst)}. It may be difficult to "
            "use and may have major changes at any time."
        )

    direct = f"{decl.__qualname__} {description(False)}"
    rst = f"""\
.. warning::

   ``{decl.__qualname__}`` {description(True)}
"""

    return direct, rst


def experimental(*, reason, issues=None):
    """
    A decorator to mark a function, method or class as experimental, meaning it may not be complete.

    Args:
        reason (str): why this is experimental
        issues (list of int, optional): any relevant ``stellargraph/stellargraph`` issues
    """
    if issues is None:
        issues = []

    def decorator(decl):
        # add warning at the start of the documentation
        # <https://docutils.sourceforge.io/docs/ref/rst/directives.html#caution>
        direct_msg, rst_msg = messages(decl, reason, issues)
        if decl.__doc__ is not None:
            decl.__doc__ = f"{rst_msg}\n\n{dedent(decl.__doc__)}"
        else:
            decl.__doc__ = rst_msg

        is_class = isinstance(decl, type)

        func_to_wrap = decl.__init__ if is_class else decl

        @functools.wraps(func_to_wrap)
        def new_func(*args, **kwargs):
            warnings.warn(direct_msg, ExperimentalWarning, stacklevel=2)
            return func_to_wrap(*args, **kwargs)

        if is_class:
            decl.__init__ = new_func
            return decl
        else:
            return new_func

    return decorator
