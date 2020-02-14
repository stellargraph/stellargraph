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

import pytest
import random
from stellargraph.core.experimental import experimental, ExperimentalWarning

# some random data to check args are being passed through correctly
@pytest.fixture
def args():
    return random.random(), random.random()


@pytest.fixture
def kwargs():
    return {str(random.random()): random.random()}


@experimental(reason="function is experimental", issues=[123, 456])
def func(*args, **kwargs):
    return args, kwargs


def test_experimental_func(args, kwargs):
    with pytest.warns(
        ExperimentalWarning,
        match=r"^func is experimental: function is experimental \(see: .*/123, .*/456\)\.",
    ):
        ret = func(*args, **kwargs)

    assert ret == (args, kwargs)


@experimental(reason="class is experimental", issues=[])
class ClassNoInit:
    pass


@experimental(reason="class is experimental")
class ClassInit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_experimental_class(args, kwargs):
    with pytest.warns(
        ExperimentalWarning,
        match=r"^ClassNoInit is experimental: class is experimental\.",
    ):
        ClassNoInit()

    with pytest.warns(
        ExperimentalWarning,
        match=r"^ClassInit is experimental: class is experimental\.",
    ):
        instance = ClassInit(*args, **kwargs)

    assert instance.args == args
    assert instance.kwargs == kwargs


class Class:
    @experimental(reason="method is experimental")
    def method(self, *args, **kwargs):
        return self, args, kwargs


def test_experimental_method(args, kwargs):
    instance = Class()
    with pytest.warns(
        ExperimentalWarning,
        match=r"^Class\.method is experimental: method is experimental\.",
    ):
        ret = instance.method(*args, **kwargs)

    assert ret[0] is instance
    assert ret[1:] == (args, kwargs)
