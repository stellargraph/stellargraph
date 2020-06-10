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

__all__ = [
    "poincare_ball_distance",
    "poincare_ball_exp",
    "poincare_ball_mobius_add",
]

import tensorflow as tf
import numpy as np


# helper functions to manage numerical issues, inspired by https://github.com/dalab/hyperbolic_nn

PROJECTION_EPS = 1e-5
TANH_LIMIT = 15.0
ATANH_LIMIT = tf.math.nextafter(1, 0)


def _project(c, x):
    """
    Ensure ``x`` lies on the Poincaré ball with curvature ``-c``, in the presence of small numerical
    errors.
    """
    max_norm = tf.math.rsqrt(c) * (1 - PROJECTION_EPS)
    return tf.clip_by_norm(x, clip_norm=max_norm, axes=-1)


def _tanh(x):
    return tf.tanh(tf.clip_by_value(x, -TANH_LIMIT, TANH_LIMIT))


def _atanh(x):
    return tf.atanh(tf.clip_by_value(x, -ATANH_LIMIT, ATANH_LIMIT))


def poincare_ball_mobius_add(c, x, y):
    r"""
    Möbius addition of ``x`` and ``y``, on the Poincaré ball with curvature ``-c``: :math:`\mathbf{x} \oplus^c \mathbf{y}`.

    See Section 2 of [1] for more details.

    [1] O.-E. Ganea, G. Bécigneul, and T. Hofmann, “Hyperbolic Neural Networks,” `arXiv:1805.09112 <http://arxiv.org/abs/1805.09112>`_, Jun. 2018.

    Args:
        c (tensorflow Tensor-like): the curvature of the hyperbolic space(s). Must be able to be
            broadcast to ``x`` and ``y``.
        x (tensorflow Tensor-like): a tensor containing vectors in hyperbolic space, where each
            vector is an element of the last axis (for example, if ``x`` has shape ``(2, 3, 4)``, it
            represents ``2 * 3 = 6`` hyperbolic vectors, each of length ``4``). Must be able to be
            broadcast to ``y``.
        y (tensorflow Tensor-like): a tensor containing vectors in hyperbolic space, where each
            vector is an element of the last axis similar to ``x``. Must be able to be broadcast to
            ``x``.

    Returns:
        A TensorFlow Tensor containing the Möbius addition of each of the vectors (last axis) in
        ``x`` and ``y``, using the corresponding curvature from ``c``. This tensor has the same
        shape as the Euclidean equivalent ``x + y``.
    """
    x_norm2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    y_norm2 = tf.reduce_sum(y * y, axis=-1, keepdims=True)
    x_dot_y = tf.reduce_sum(x * y, axis=-1, keepdims=True)

    inner = 1 + 2 * c * x_dot_y
    numer = (inner + c * y_norm2) * x + (1 - c * x_norm2) * y
    denom = inner + c * c * x_norm2 * y_norm2

    return _project(c, numer / denom)


def poincare_ball_exp(c, x, v):
    r"""
    The exponential map of ``v`` at ``x`` on the Poincaré ball with curvature ``-c``:
    :math:`\exp_{\mathbf{x}}^c(\mathbf{v})`.

    See Section 2 of [1] for more details.

    [1] O.-E. Ganea, G. Bécigneul, and T. Hofmann, “Hyperbolic Neural Networks,” `arXiv:1805.09112 <http://arxiv.org/abs/1805.09112>`_, Jun. 2018.

    Args:
        c (tensorflow Tensor-like): the curvature of the hyperbolic space(s). Must be able to be
            broadcast to ``x`` and ``v``.

        x (tensorflow Tensor-like, optional): a tensor containing vectors in hyperbolic space
            representing the base points for the exponential map, where each vector is an element of
            the last axis (for example, if ``x`` has shape ``(2, 3, 4)``, it represents ``2 * 3 =
            6`` hyperbolic vectors, each of length ``4``). Must be able to be broadcast to ``v``. An
            explicit ``x = None`` is equivalent to ``x`` being all zeros, but uses a more efficient
            form of :math:`\exp_{\mathbf{0}}^c(\mathbf{v})`.

        v (tensorflow Tensor-like): a tensor containing vectors in Euclidean space representing the
            tangent vectors for the exponential map, where each vector is an element of the last
            axis similar to ``x``. Must be able to be broadcast to ``x``.
    """

    v_norm2 = tf.reduce_sum(v * v, axis=-1, keepdims=True)
    c_v_norm = tf.sqrt(c * v_norm2)

    if x is None:
        coeff = _tanh(c_v_norm) / c_v_norm
        return _project(c, coeff * v)

    x_norm2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    inner = c_v_norm / (1 - c * x_norm2)
    coeff = _tanh(inner) / c_v_norm
    return poincare_ball_mobius_add(c, x, coeff * v)


def poincare_ball_distance(c, x, y):
    r"""
    Distance between ``x`` and ``y``, on the Poincaré ball with curvature ``-c``: :math:`d_c(\mathbf{x}, \mathbf{y})`.

    See Section 2 of [1] for more details.

    [1] O.-E. Ganea, G. Bécigneul, and T. Hofmann, “Hyperbolic Neural Networks,” `arXiv:1805.09112 <http://arxiv.org/abs/1805.09112>`_, Jun. 2018.

    Args:
        c (tensorflow Tensor-like): the curvature of the hyperbolic space(s). Must be able to be
            broadcast to ``x`` and ``y``.
        x (tensorflow Tensor-like): a tensor containing vectors in hyperbolic space, where each
            vector is an element of the last axis (for example, if ``x`` has shape ``(2, 3, 4)``, it
            represents ``2 * 3 = 6`` hyperbolic vectors, each of length ``4``). Must be able to be
            broadcast to ``y``.
        y (tensorflow Tensor-like): a tensor containing vectors in hyperbolic space, where each
            vector is an element of the last axis similar to ``x``. Must be able to be broadcast to
            ``x``.

    Returns:
        A TensorFlow Tensor containing the hyperbolic distance between each of the vectors (last
        axis) in ``x`` and ``y``, using the corresponding curvature from ``c``. This tensor has the
        same shape as the Euclidean equivalent ``tf.norm(x - y)``.
    """
    sqrt_c = tf.sqrt(c)
    return (2 / sqrt_c) * _atanh(
        sqrt_c * tf.norm(poincare_ball_mobius_add(c, -x, y), axis=-1)
    )
