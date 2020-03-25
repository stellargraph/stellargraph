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
import numpy as np


def separated(values, *, limit, stringify, sep):
    """
    Print up to ``limit`` values with a separator.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
        sep (str): the separator to use between elements (and the "... (NNN more)" continuation)
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation


def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    return separated(values, limit=limit, stringify=stringify, sep=", ")


def require_dataframe_has_columns(name, df, columns):
    if not set(columns).issubset(df.columns):
        raise ValueError(
            f"{name}: expected {comma_sep(columns)} columns, found: {comma_sep(df.columns)}"
        )


def require_integer_in_range(x, variable_name, min_val=-np.inf, max_val=np.inf):
    """
    A function to verify that a variable is an integer in a specified closed range.
    Args:
        x: the variable to check
        variable_name (str): the name of the variable to print out in error messages
        min_val: the minimum value that `x` can attain
        min_val: the maximum value that `x` can attain
    """

    if not isinstance(x, int):
        raise TypeError(f"{variable_name}: expected int, found {type(x).__name__}")

    if x < min_val or x > max_val:

        if min_val == -np.inf:
            region = f"<= {max_val}"
        elif max_val == np.inf:
            region = f">= {min_val}"
        else:
            region = f"in the range [{min_val}, {max_val}]"

        raise ValueError(f"{variable_name}: expected integer {region}, found {x}")
