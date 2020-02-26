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

import itertools

import pytest

import pandas as pd
import numpy as np

from tensorflow.keras import Model, initializers, losses

from stellargraph.mapper.knowledge_graph import KGTripleGenerator
from stellargraph.layer.knowledge_graph import ComplEx, DistMult

from .. import test_utils
from ..test_utils.graphs import knowledge_graph


pytestmark = [
    test_utils.ignore_stellargraph_experimental_mark,
    pytest.mark.filterwarnings(
        r"ignore:ComplEx:stellargraph.core.experimental.ExperimentalWarning"
    ),
]


def triple_df(*values):
    return pd.DataFrame(values, columns=["source", "label", "target"])


def test_complex(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large positive range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    x_inp, x_out = ComplEx(gen, 5, embeddings_initializer=init).build()

    model = Model(x_inp, x_out)
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True))

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # check the model can be trained on a few (uneven) batches
    model.fit(
        gen.flow(df.iloc[:7], negative_samples=2),
        validation_data=gen.flow(df.iloc[7:14], negative_samples=3),
    )

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the Re(<e_s, w_r, conj(e_o)>) inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = ComplEx.embeddings(model)
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o.conj()).sum(axis=1).real

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)


def test_dismult(knowledge_graph):
    # this test creates a random untrained model and predicts every possible edge in the graph, and
    # compares that to a direct implementation of the scoring method in the paper
    gen = KGTripleGenerator(knowledge_graph, 3)

    # use a random initializer with a large range, so that any differences are obvious
    init = initializers.RandomUniform(-1, 1)
    x_inp, x_out = DistMult(gen, 5, embeddings_initializer=init).build()

    model = Model(x_inp, x_out)

    every_edge = itertools.product(
        knowledge_graph.nodes(),
        knowledge_graph._edges.types.pandas_index,
        knowledge_graph.nodes(),
    )
    df = triple_df(*every_edge)

    # compute the exact values based on the model by extracting the embeddings for each element and
    # doing the y_(e_1)^T M_r y_(e_2) = <e_1, w_r, e_2> inner product
    s_idx = knowledge_graph._get_index_for_nodes(df.source)
    r_idx = knowledge_graph._edges.types.to_iloc(df.label)
    o_idx = knowledge_graph._get_index_for_nodes(df.target)

    nodes, edge_types = DistMult.embeddings(model)
    # the rows correspond to the embeddings for the given edge, so we can do bulk operations
    e_s = nodes[s_idx, :]
    w_r = edge_types[r_idx, :]
    e_o = nodes[o_idx, :]
    actual = (e_s * w_r * e_o).sum(axis=1)

    # predict every edge using the model
    prediction = model.predict(gen.flow(df))

    # (use an absolute tolerance to allow for catastrophic cancellation around very small values)
    assert np.allclose(prediction[:, 0], actual, rtol=1e-3, atol=1e-14)
