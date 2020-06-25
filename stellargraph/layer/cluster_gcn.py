# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
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

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape
from .misc import deprecated_model_function, GatherIndices
from ..mapper import ClusterNodeGenerator
from .gcn import GraphConvolution, GCN

import warnings


class ClusterGraphConvolution(GraphConvolution):
    """
    Deprecated: use :class:`.GraphConvolution`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ClusterGraphConvolution has been replaced by GraphConvolution without functionality change",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ClusterGCN(GCN):
    """
    Deprecated: use :class:`stellargraph.layer.GCN` with :class:`stellargraph.mapper.ClusterNodeGenerator`.
    """

    def __init__(
        self,
        # the parameter order is slightly different between this and GCN, so the *args,
        # **kwargs trick doesn't work
        layer_sizes,
        activations,
        generator,
        bias=True,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
    ):
        warnings.warn(
            "ClusterGCN has been replaced by GCN with little functionality change (the GCN class removes the batch dimension in some cases)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            layer_sizes=layer_sizes,
            generator=generator,
            bias=bias,
            dropout=dropout,
            activations=activations,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            # for compatibility
            squeeze_output_batch=False,
        )
