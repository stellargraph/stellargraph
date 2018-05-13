from stellar.layer.hinsage import *
import keras
import numpy as np
import pytest


def test_mean_hin_agg_constructor():
    agg = MeanHinAggregator(output_dim=2)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert not agg.has_bias
    assert agg.act == keras.backend.relu


def test_mean_hin_agg_constructor_1():
    agg = MeanHinAggregator(output_dim=2, bias=True, act=lambda x: x + 1)
    assert agg.output_dim == 2
    assert agg.half_output_dim == 1
    assert agg.has_bias
    assert agg.act(2) == 3


def test_mean_hin_agg_apply():
    agg = MeanHinAggregator(2, act=lambda z: z)
    agg._initializer = 'ones'
    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(1, 2, 2)),
        keras.Input(shape=(1, 2, 4))
    ]
    out = agg(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[[2, 2], [2, 2]]]]),
        np.array([[[[3, 3, 3, 3], [3, 3, 3, 3]]]])
    ]

    actual = model.predict(x)
    print(actual)
    expected = np.array([[[2, 8]]])
    assert actual == pytest.approx(expected)


def test_hinsage_constructor():
    hs = Hinsage(
        output_dims=[{'1': 2, '2': 2}, {'1': 2}],
        n_samples=[2, 2],
        input_neigh_tree=[('1', [1, 2]), ('1', [3, 4]), ('2', [5]), ('1', []), ('2', []), ('2', [])],
        input_dim={'1': 2, '2': 2}
    )
    assert hs.n_layers == 2
    assert hs.n_samples == [2, 2]
    assert not hs.bias


def test_hinsage_apply():
    hs = Hinsage(
        output_dims=[{'1': 2, '2': 2}, {'1': 2}],
        n_samples=[2, 2],
        input_neigh_tree=[('1', [1, 2]), ('1', [3, 4]), ('2', [5]), ('1', []), ('2', []), ('2', [])],
        input_dim={'1': 2, '2': 4}
    )
    hs._normalization = lambda z: z
    for aggs in hs._aggs:
        for _, agg in aggs.items():
            agg._initializer = 'ones'

    inp = [
        keras.Input(shape=(1, 2)),
        keras.Input(shape=(2, 2)),
        keras.Input(shape=(2, 4)),
        keras.Input(shape=(4, 2)),
        keras.Input(shape=(4, 4)),
        keras.Input(shape=(4, 4))
    ]

    out = hs(inp)
    model = keras.Model(inputs=inp, outputs=out)

    x = [
        np.array([[[1, 1]]]),
        np.array([[[2, 2], [2, 2]]]),
        np.array([[[4, 4, 4, 4], [4, 4, 4, 4]]]),
        np.array([[[3, 3], [3, 3], [3, 3], [3, 3]]]),
        np.array([[[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]]),
        np.array([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]])
    ]

    actual = model.predict(x)
    expected = np.array([[[12, 35.5]]])
    assert actual == pytest.approx(expected)
