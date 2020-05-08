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
    "poincare_ball_exp_map",
    "poincare_ball_exp_map0",
    "poincare_ball_mobius_add",
]

import tensorflow as tf


def poincare_ball_mobius_add(c, x, y):
    r"""
    Möbius addition of ``x`` and ``y``, on the Poincaré ball with curvature ``c``: :math:`\mathbf{x} \oplus^c \mathbf{y}`.
    """
    x_norm2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    y_norm2 = tf.reduce_sum(y * y, axis=-1, keepdims=True)
    x_dot_y = tf.reduce_sum(x * y, axis=-1, keepdims=True)

    inner = 1 + 2 * c * x_dot_y
    numer = (inner + c * y_norm2) * x + (1 - c * x_norm2) * y
    denom = inner + c * c * x_norm2 * y_norm2

    return numer / denom


def _exp_map_multiply(c, v, denom):
    v_norm2 = tf.reduce_sum(v * v, axis=-1, keepdims=True)
    c_v_norm = tf.sqrt(c * v_norm2)
    inner = c_v_norm if denom is None else c_v_norm / denom
    coeff = tf.math.tanh(inner) / c_v_norm
    return coeff * v


def poincare_ball_exp_map(x, c, v):
    r"""
    The exponential map of ``v`` at ``x`` on the Poincaré ball with curvature ``c``:
    :math:`\exp_{\mathbf{x}}^c(\mathbf{v})`.
    """

    x_norm2 = tf.reduce_sum(x * x, axis=-1, keepdims=True)
    return poincare_ball_mobius_add(c, x, _exp_map_multiply(c, v, 1 - c * x_norm2))


def poincare_ball_exp_map0(c, v):
    r"""
    :math:`\exp_{\mathbf{x}}^c(\mathbf{v})` specialised for :math:`\mathbf{x} = \mathbf{0}`.

    .. seealso:: :func:`poincare_ball_exp_map`
    """
    return _exp_map_multiply(c, v, denom=None)


def poincare_ball_distance(c, x, y):
    """
    Distance between ``x`` and ``y``, on the Poincaré ball with curvature ``c``: :math:`d_c(\mathbf{x}, \mathbf{y})`.
    """
    sqrt_c = tf.sqrt(c)
    return (2 / sqrt_c) * tf.atanh(
        sqrt_c * tf.norm(-poincare_ball_mobius_add(c, x, y), axis=-1)
    )
