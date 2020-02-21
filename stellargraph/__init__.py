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

"""
Stellar Machine Learning Library

"""

__all__ = [
    "data",
    "layer",
    "mapper",
    "utils",
    "StellarDiGraph",
    "StellarGraph",
    "__version__",
]

# Version
from .version import __version__

# Import modules
from stellargraph import mapper, layer, utils

# Top-level imports
from stellargraph.core.graph import StellarGraph, StellarDiGraph
from stellargraph.core.schema import GraphSchema
from stellargraph.utils.calibration import TemperatureCalibration, IsotonicCalibration
from stellargraph.utils.calibration import (
    plot_reliability_diagram,
    expected_calibration_error,
)
from stellargraph.utils.ensemble import Ensemble, BaggingEnsemble

# Custom layers for keras deserialization:
custom_keras_layers = {
    "GraphConvolution": layer.GraphConvolution,
    "ClusterGraphConvolution": layer.ClusterGraphConvolution,
    "GraphAttention": layer.GraphAttention,
    "GraphAttentionSparse": layer.GraphAttentionSparse,
    "SqueezedSparseConversion": layer.SqueezedSparseConversion,
    "MeanAggregator": layer.graphsage.MeanAggregator,
    "MaxPoolingAggregator": layer.graphsage.MaxPoolingAggregator,
    "MeanPoolingAggregator": layer.graphsage.MeanPoolingAggregator,
    "AttentionalAggregator": layer.graphsage.AttentionalAggregator,
    "MeanHinAggregator": layer.hinsage.MeanHinAggregator,
    "RelationalGraphConvolution": layer.rgcn.RelationalGraphConvolution,
    "PPNPPropagationLayer": layer.ppnp.PPNPPropagationLayer,
    "APPNPPropagationLayer": layer.appnp.APPNPPropagationLayer,
}
