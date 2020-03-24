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


__all__ = [
    "data",
    "datasets",
    "calibration",
    "ensemble",
    "interpretability",
    "losses",
    "layer",
    "mapper",
    "utils",
    "custom_keras_layers",
    "StellarDiGraph",
    "StellarGraph",
    "GraphSchema",
    "__version__",
]

# Version
from .version import __version__

# Import modules
from stellargraph import (
    data,
    calibration,
    datasets,
    ensemble,
    interpretability,
    losses,
    layer,
    mapper,
    utils,
)

# Top-level imports
from stellargraph.core.graph import StellarGraph, StellarDiGraph
from stellargraph.core.schema import GraphSchema
import warnings

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


def _top_level_deprecation_warning(name, path):
    warnings.warn(
        f"'{name}' is no longer available at the top-level. "
        f"Please use 'stellargraph.{path}.{name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def expected_calibration_error(*args, **kwargs):
    _top_level_deprecation_warning("expected_calibration_error", "calibration")
    return calibration.expected_calibration_error(*args, **kwargs)


def plot_reliability_diagram(*args, **kwargs):
    _top_level_deprecation_warning("plot_reliability_diagram", "calibration")
    return calibration.plot_reliability_diagram(*args, **kwargs)


def Ensemble(*args, **kwargs):
    _top_level_deprecation_warning("Ensemble", "ensemble")
    return ensemble.Ensemble(*args, **kwargs)


def BaggingEnsemble(*args, **kwargs):
    _top_level_deprecation_warning("BaggingEnsemble", "ensemble")
    return ensemble.BaggingEnsemble(*args, **kwargs)


def TemperatureCalibration(*args, **kwargs):
    _top_level_deprecation_warning("TemperatureCalibration", "calibration")
    return calibration.TemperatureCalibration(*args, **kwargs)


def IsotonicCalibration(*args, **kwargs):
    _top_level_deprecation_warning("IsotonicCalibration", "calibration")
    return calibration.IsotonicCalibration(*args, **kwargs)
