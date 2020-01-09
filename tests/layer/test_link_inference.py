# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
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

"""
Tests for link inference functions
"""

from stellargraph.layer.link_inference import *
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pytest


class Test_Link_Inference(object):
    """
    Group of tests for link_inference() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link inference output

    def test_ip(self):
        """ Test the 'ip' binary operator on orthogonal vectors"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        expected = np.dot(x_src, x_dst)

        x_src = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_inference(edge_embedding_method="ip", output_act="linear")(
            [x_src, x_dst]
        )
        print(
            "link inference with 'ip' operator on orthonormal vectors: {}, expected: {}".format(
                li, expected
            )
        )
        assert li.numpy() == pytest.approx(expected, abs=1.5e-7)
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = link_inference(edge_embedding_method="ip", output_act="linear")(
            [x_src, x_src]
        )
        print("link inference with 'ip' operator on unit vector: ", li)
        assert li.numpy() == pytest.approx(1, abs=1.5e-7)

        # Test sigmoid activation
        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")(
            [x_src, x_dst]
        )
        assert li.numpy() == pytest.approx(0.5)

        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")(
            [x_src, x_src]
        )
        assert li.numpy() == pytest.approx(0.7310586, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'hadamard', 'l1', 'l2', 'avg'"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        x_src = x_src.reshape(1, 1, self.d)
        x_dst = x_dst.reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = link_inference(output_dim=self.d_out, edge_embedding_method=op)(
                [inp_src, inp_dst]
            )
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            print("link inference with '{}' operator: {}".format(op, res.flatten()))

            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)


class Test_Link_Classification(object):
    """
    Group of tests for link_classification() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link classification output

    def test_ip(self):
        """ Test the 'ip' binary operator on orthogonal vectors"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        expected = np.dot(x_src, x_dst)

        x_src = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        # Test linear activation
        li = link_classification(edge_embedding_method="ip", output_act="linear")(
            [x_src, x_dst]
        )
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = link_classification(edge_embedding_method="ip", output_act="linear")(
            [x_src, x_src]
        )
        assert li.numpy()[0, 0] == pytest.approx(1, abs=1.5e-7)

        # Test sigmoid activation
        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")(
            [x_src, x_dst]
        )
        assert li.numpy() == pytest.approx(0.5, abs=1.5e-7)

        li = link_classification(edge_embedding_method="ip", output_act="sigmoid")(
            [x_src, x_src]
        )
        assert li.numpy() == pytest.approx(0.7310586, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'hadamard', 'l1', 'l2', 'avg'"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        x_src = x_src.reshape(1, 1, self.d)
        x_dst = x_dst.reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = link_classification(output_dim=self.d_out, edge_embedding_method=op)(
                [inp_src, inp_dst]
            )
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            print(
                "link classification with '{}' operator: {}".format(op, res.flatten())
            )

            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)
            assert all(res.flatten() >= 0)
            assert all(res.flatten() <= 1)


class Test_Link_Regression(object):
    """
    Group of tests for link_regression() function
    """

    d = 100  # dimensionality of embedding vector space
    d_out = 10  # dimensionality of link classification output
    clip_limits = (0, 1)

    def test_ip(self):
        """ Test the 'ip' binary operator on orthogonal vectors"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        expected = np.dot(x_src, x_dst)

        x_src = tf.constant(x_src, shape=(1, self.d), dtype="float64")
        x_dst = tf.constant(x_dst, shape=(1, self.d), dtype="float64")

        li = link_regression(edge_embedding_method="ip")([x_src, x_dst])
        print(
            "link regression with 'ip' operator on orthonormal vectors: {}, expected: {}".format(
                li, expected
            )
        )
        assert li.numpy() == pytest.approx(0, abs=1.5e-7)

        li = link_regression(edge_embedding_method="ip")([x_src, x_src])
        print("link regression with 'ip' operator on unit vector: ", li)
        assert li.numpy() == pytest.approx(1, abs=1.5e-7)

    def test_mul_l1_l2_avg(self):
        """ Test the binary operators: 'mul'/'hadamard', 'l1', 'l2', 'avg'"""

        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        x_src = x_src.reshape(1, 1, self.d)
        x_dst = x_dst.reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = link_regression(output_dim=self.d_out, edge_embedding_method=op)(
                [inp_src, inp_dst]
            )
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            print("link regression with '{}' operator: {}".format(op, res.flatten()))

            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)

    def test_clip_limits(self):
        """
        Test calling with the leaky clip thresholds
        Not sure what a meaningful test should do (as the LeakyClippedLinear layer provides some advantages at model training),
        so just making sure applying the clip limits doesn't break anything.
        """

        print("\n Testing clip limits...")
        x_src = np.random.randn(self.d)
        x_src /= np.linalg.norm(x_src)  # normalize x_src
        x_dst = np.random.randn(self.d)
        x_dst -= x_dst.dot(x_src) * x_src  # make x_dst orthogonal to x_src
        x_dst /= np.linalg.norm(x_dst)  # normalize x_dst

        x_src = x_src.reshape(1, 1, self.d)
        x_dst = x_dst.reshape(1, 1, self.d)

        inp_src = keras.Input(shape=(1, self.d))
        inp_dst = keras.Input(shape=(1, self.d))

        for op in ["mul", "l1", "l2", "avg"]:
            out = link_regression(
                output_dim=self.d_out,
                edge_embedding_method=op,
                clip_limits=self.clip_limits,
            )([inp_src, inp_dst])
            li = keras.Model(inputs=[inp_src, inp_dst], outputs=out)

            res = li.predict(x=[x_src, x_dst])
            print("link regression with '{}' operator: {}".format(op, res.flatten()))

            assert res.shape == (1, self.d_out)
            assert isinstance(res.flatten()[0], np.float32)
