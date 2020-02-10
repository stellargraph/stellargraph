# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
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

import abc

import numpy as np

class ColumnsBase(abc.ABC):
    def __init__(self, columns):
        self._columns = columns

    @property
    def column_names(self):
        return self._columns.keys()

    @abc.abstractmethod
    def column(self, name): ...

    @abc.abstractmethod
    def columns(self, *names): ...

    @abc.abstractmethod
    def add_columns(self, new_columns): ...

    def _drop_columns_dict(self, to_drop):
        to_drop = set(to_drop)
        return {name: data for name, data in self._columns.items() if name not in to_drop}

    @abc.abstractmethod
    def drop_columns(self, *to_drop): ...

    def _select_columns_dict(self, to_keep):
        return {name: self._columns[name] for name in to_keep}

    @abc.abstractmethod
    def select_columns(self, *to_keep): ...

    @abc.abstractmethod
    def select_rows(self, selector): ...

    @abc.abstractmethod
    def iter_rows(self, *columns): ...

class Columns(ColumnsBase):
    """
    Columns is a very basic columnar data store, that is much less flexible and thus much faster
    than pandas DataFrames.

    Args:
        columns (dict of hashable to ndarray): a map from column name to column values (all lengths
            must be the same)
        unsafe_unchecked (bool, optional): if True, this does not do any checks for validity, and
            invariants must be maintained by the caller
    """

    def __init__(self, columns, unsafe_unchecked=False):
        if not unsafe_unchecked:
            if not isinstance(columns, dict):
                raise TypeError(f"columns: expected dict, found {type(columns)}")

            length = None
            for c, v in columns.items():
                if not isinstance(v, np.ndarray):
                    raise TypeError(
                        f"columns[{c!r}]: expected ndarray, found {type(columns)}"
                    )
                if length is None:
                    length = len(v)
                elif len(v) != length:
                    raise ValueError(
                        f"columns[{c!r}]: expected length {length} to match other columns, found {len(v)}"
                    )

        self._columns = columns

    @property
    def lazy(self) -> "LazyColumns":
        return LazyColumns(self)

    @property
    def column_names(self):
        return self._columns.keys()

    def column(self, name):
        return self._columns[name]

    def columns(self, *names):
        return tuple(self._columns[name] for name in names)

    def add_columns(self, new_columns):
        return Columns({**self._columns, **new_columns})

    def drop_columns(self, *to_drop):
        return Columns(self._drop_columns_dict(to_drop) unsafe_unchecked=True)

    def select_columns(self, *to_keep):
        return Columns(self._select_columns_dict(to_keep), unsafe_unchecked=True)

    def select_rows(self, selector) -> "Columns":
        """
        Select a subset of the rows (values from each column).

        Args:
            selector (any valid indexer for an ndarray): the rows to select
        """

        new_cols = {name: data[selector] for name, data in self._columns.items()}
        # we've sliced each column with the same indices, so they'll match
        return Columns(new_cols, unsafe_unchecked=True)

    def iter_rows(self, *columns):
        """
        Iterate over tuples of the columns specified.

        Args:

        """
        return zip(*(self.column(name) for name in columns))


class LazyColumns(ColumnsBase):
    def __init__(self, columns=None, unsafe_columns_dict=None, row_selectors=[]):
        if columns is not None:
            if not isinstance(columns, Columns):
                raise TypeError(f"columns: expected Columns, found {type(columns)}")

            columns_dict = self._columns
        else:
            if not isinstance(unsafe_columns_dict, dict):
                raise TypeError("unsafe_columns_dict: expected dict, found {type(unsafe_columns_dict)}")
            columns_dict = unsafe_columns_dict

        super().__init__(columns_dict)
        self._row_selectors = row_selectors

    @cached_property
    def force(self) -> "Columns":
        c = self._columns.copy()
        for sel in self._row_selectors:
            for k, v in c.items():
                c[k] = v[sel]
        return Columns(c, unsafe_unchecked=True)

    def _nonlazy_method(self, method_name):
        raise TypeError(f"'{method_name}' can only be applied to 'Columns', found 'LazyColumns'; use '.force' to apply lazy transformations and create a 'Columns'")

    def column(self, name):
        self._nonlazy_method("column")

    def columns(self, *names):
        self._nonlazy_method("columns")

    def add_columns(self, new_columns):
        self._nonlazy_method("add_columns")

    def drop_columns(self, *to_drop):
        return LazyColumns(unsafe_columns_dict=self._drop_columns_dict(to_drop), self.row_selectors)

    def select_columns(self, *to_keep):
        return LazyColumns(unsafe_columns_dict=self._select_columns_dict(to_drop), self.row_selectors)

    def select_rows(self, selector):
        new_selectors = self.row_selectors.copy()
        new_selectors.append(selector)
        return LazyColumns(unsafe_columns_dict=self._columns, new_selectors)

    @abc.abstractmethod
    def iter_rows(self, *columns):
        self._nonlazy_method("iter_rows")
