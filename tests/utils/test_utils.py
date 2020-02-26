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

import importlib

import pytest
import stellargraph
from stellargraph import utils


def test_moves_smoke():
    # do a check without all the getattr trickery, to validate that there's nothing special about
    # that
    with pytest.warns(None) as record:
        # no warning
        utils.plot_history

        # deprecated warning
        old = utils.BaggingEnsemble

    assert old is stellargraph.ensemble.BaggingEnsemble

    assert len(record) == 1
    assert (
        str(record.pop(DeprecationWarning).message)
        == "'stellargraph.utils.BaggingEnsemble' has been moved to 'stellargraph.ensemble.BaggingEnsemble'"
    )


# a map from the new `stellargraph.XYZ` module to the base names that that module contains, where
# each name ABC was previously available at `stellargraph.utils.ABC`.
_MOVED = {
    "ensemble": [None, "BaggingEnsemble", "Ensemble"],
    "interpretability": [
        "saliency_maps",
        "GradientSaliencyGAT",
        "IntegratedGradients",
        "IntegratedGradientsGAT",
    ],
    "interpretability.saliency_maps": [
        "integrated_gradients",
        "integrated_gradients_gat",
        "saliency_gat",
    ],
    "calibration": [
        None,
        "IsotonicCalibration",
        "TemperatureCalibration",
        "expected_calibration_error",
        "plot_reliability_diagram",
    ],
}


@pytest.mark.parametrize(
    "new_module_name,item_name",
    [(new_module, name) for new_module, names in _MOVED.items() for name in names],
)
def test_moves_all_top_level_items(new_module_name, item_name):
    new_module_location = f"stellargraph.{new_module_name}"
    new_module = importlib.import_module(new_module_location)

    if item_name is None:
        item_name = new_module_name
        new_location = new_module_location
        new_value = new_module
    else:
        new_location = f"{new_module_location}.{item_name}"
        new_value = getattr(new_module, item_name)

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.{item_name}' has been moved to '{new_location}'",
    ):
        old_value = getattr(utils, item_name)

    assert old_value is new_value


def test_moves_submodule_import():
    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.calibration' has been moved to 'stellargraph.calibration'",
    ):
        import stellargraph.utils.calibration as c

        assert c.IsotonicCalibration is stellargraph.calibration.IsotonicCalibration

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.ensemble' has been moved to 'stellargraph.ensemble'",
    ):
        # test a 'from ... import' import
        from stellargraph.utils import ensemble

        assert ensemble.Ensemble is stellargraph.ensemble.Ensemble

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.saliency_maps' has been moved to 'stellargraph.interpretability.saliency_maps'",
    ):
        import stellargraph.utils.saliency_maps as sm

        assert (
            sm.GradientSaliencyGAT is stellargraph.interpretability.GradientSaliencyGAT
        )

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.saliency_maps.integrated_gradients' has been moved to 'stellargraph.interpretability.saliency_maps.integrated_gradients'",
    ):
        import stellargraph.utils.saliency_maps.integrated_gradients as ig

        assert (
            ig.IntegratedGradients is stellargraph.interpretability.IntegratedGradients
        )

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.saliency_maps.integrated_gradients_gat' has been moved to 'stellargraph.interpretability.saliency_maps.integrated_gradients_gat'",
    ):
        import stellargraph.utils.saliency_maps.integrated_gradients_gat as ig_gat

        assert (
            ig_gat.IntegratedGradientsGAT
            is stellargraph.interpretability.IntegratedGradientsGAT
        )

    with pytest.warns(
        DeprecationWarning,
        match=f"'stellargraph.utils.saliency_maps.saliency_gat' has been moved to 'stellargraph.interpretability.saliency_maps.saliency_gat'",
    ):
        import stellargraph.utils.saliency_maps.saliency_gat as s_gat

        assert (
            s_gat.GradientSaliencyGAT
            is stellargraph.interpretability.GradientSaliencyGAT
        )
