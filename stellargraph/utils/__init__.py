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
This contains the utility objects used by the StellarGraph library.

"""

import warnings
import sys
import types

from .. import calibration, ensemble
from ..calibration import *
from ..ensemble import *
from ..interpretability import *
from .history import *

_E = "ensemble"
_C = "calibration"
_I = "interpretability"

_MAPPING = {
    # modules
    "calibration": (None, calibration),
    "ensemble": (None, ensemble),
    "saliency_maps": (None, saliency_maps),
    "integrated_gradients": (None, saliency_maps.integrated_gradients),
    "integrated_gradients_gat": (None, saliency_maps.integrated_gradients_gat),
    "saliency_gat": (None, saliency_maps.saliency_gat),
    # calibration
    "IsotonicCalibration": (_C, IsotonicCalibration),
    "TemperatureCalibration": (_C, TemperatureCalibration),
    "expected_calibration_error": (_C, expected_calibration_error),
    "plot_reliability_diagram": (_C, plot_reliability_diagram),
    # ensembles
    "Ensemble": (_E, Ensemble),
    "BaggingEnsemble": (_E, BaggingEnsemble),
    # interpretability
    "IntegratedGradients": (_I, IntegratedGradients),
    "IntegratedGradientsGAT": (_I, IntegratedGradientsGAT),
    "GradientSaliencyGAT": (_I, GradientSaliencyGAT),
}


class _Wrapper(types.ModuleType):
    def __init__(self, current):
        super().__init__(current.__name__, current.__doc__)
        self.__package__ = current.__package__
        self.__loader__ = current.__loader__
        self.__path__ = current.__path__

    plot_history = staticmethod(plot_history)

    def __getattr__(self, attr):
        try:
            new_module_name, new_value = _MAPPING[attr]
        except KeyError:
            # don't know about it, so do the normal access
            return super().__getattribute__(attr)
        else:
            # this attribute looks like one of the deprecated ones!
            if isinstance(new_value, types.ModuleType):
                new_location = new_value.__name__
            else:
                new_location = f"stellargraph.{new_module_name}.{new_value.__name__}"

            warnings.warn(
                f"'stellargraph.utils.{attr}' has been moved to '{new_location}'",
                DeprecationWarning,
            )
            return new_value


sys.modules[__name__] = _Wrapper(sys.modules[__name__])
