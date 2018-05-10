from stellar.layer.graphsage import *
import keras
import numpy as np
import unittest


class MeanAggregatorTest(unittest.TestCase):
    def test_constructor(self):
        agg = MeanAggregator(2)
        self.assertEqual(agg.output_dim, 2)
        self.assertEqual(agg.half_output_dim, 1)
        self.assertFalse(agg.has_bias)
        self.assertEqual(keras.backend.relu, agg.act)

    def test_constructor_1(self):
        agg = MeanAggregator(output_dim=4, bias=True, act=lambda x: x + 1)
        self.assertEqual(agg.output_dim, 4)
        self.assertEqual(agg.half_output_dim, 2)
        self.assertTrue(agg.has_bias)
        self.assertEqual(3, agg.act(2))

    def test_apply(self):
        agg = MeanAggregator(4, act=lambda x: x)
        agg._initializer = 'ones'
        inp1 = keras.Input(shape=(1, 2,))
        inp2 = keras.Input(shape=(1, 2, 2))
        out = agg([inp1, inp2])
        model = keras.Model(inputs=[inp1, inp2], outputs=out)
        x1 = np.array([[[1, 1]]])
        x2 = np.array([[[[2, 2], [3, 3]]]])
        actual = model.predict([x1, x2])
        expected = np.array([[[2, 2, 5, 5]]])
        self.assertTrue((expected == actual).all())


class GraphsageTest(unittest.TestCase):
    def test_constructor(self):
        gs = Graphsage(output_dims=[4], n_samples=[2], input_dim=2)
        self.assertEqual([2, 4], gs.dims)
        self.assertEqual([2], gs.n_samples)
        self.assertEqual(1, gs.n_layers)
        self.assertFalse(gs.bias)
        self.assertEqual(1, len(gs._aggs))

    def test_constructor_1(self):
        gs = Graphsage(output_dims=[4, 6, 8], n_samples=[2, 4, 6], input_dim=2, bias=True, dropout=0.5)
        self.assertEqual([2, 4, 6, 8], gs.dims)
        self.assertEqual([2, 4, 6], gs.n_samples)
        self.assertEqual(3, gs.n_layers)
        self.assertTrue(gs.bias)
        self.assertEqual(3, len(gs._aggs))

    def test_apply(self):
        gs = Graphsage(output_dims=[4], n_samples=[2], input_dim=2)
        gs._normalization = lambda x: x
        for agg in gs._aggs:
            agg._initializer = 'ones'

        inp1 = keras.Input(shape=(1, 2))
        inp2 = keras.Input(shape=(2, 2))
        out = gs([inp1, inp2])
        model = keras.Model(inputs=[inp1, inp2], outputs=out)

        x1 = np.array([[[1, 1]]])
        x2 = np.array([[[2, 2], [3, 3]]])

        actual = model.predict([x1, x2])
        expected = np.array([[[2, 2, 5, 5]]])
        self.assertTrue((expected == actual).all())


