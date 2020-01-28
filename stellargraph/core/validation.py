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


def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f", ... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = ", ".join(stringify(x) for x in values)
    return rendered + continuation


def require_dataframe_has_columns(name, df, columns):
    if not set(columns).issubset(df.columns):
        raise ValueError(
            f"{name}: expected {comma_sep(columns)} columns, found: {comma_sep(df.columns)}"
        )
