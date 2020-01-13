# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from stellargraph.data.converter import *
from stellargraph.utils.calibration import *


#
# Test for class TemperatureScaling
#
def test_temperature_scaling_bad_input_type():
    ts = TemperatureCalibration()

    x_train = [[1, 1], [2, 3.5]]
    y_train = [[0.9, 0.1], [0.2, 0.8]]

    x_val = [[0, 2]]
    y_val = [[0.8, 0.2]]

    # Test mixed types
    with pytest.raises(ValueError):
        ts.fit(x_train=None, y_train=np.array(y_train))

    with pytest.raises(ValueError):
        ts.fit(x_train=np.array(x_train), y_train=None)

    with pytest.raises(ValueError):
        ts.fit(x_train=x_train, y_train=np.array(y_train))

    with pytest.raises(ValueError):
        ts.fit(x_train=np.array(x_train), y_train=y_train)

    with pytest.raises(ValueError):
        ts.fit(
            x_train=np.array(x_train),
            y_train=np.array(y_train),
            x_val=np.array(x_val),
            y_val=None,
        )

    with pytest.raises(ValueError):
        ts.fit(
            x_train=np.array(x_train),
            y_train=np.array(y_train),
            x_val=None,
            y_val=np.array(y_val),
        )

    with pytest.raises(ValueError):
        ts.fit(
            x_train=np.array(x_train),
            y_train=np.array(y_train),
            x_val=np.array(x_val),
            y_val=y_val,
        )

    with pytest.raises(ValueError):
        ts.fit(
            x_train=np.array(x_train),
            y_train=np.array(y_train),
            x_val=x_val,
            y_val=np.array(y_val),
        )

    # some tests for the predict method where the input data have the wrong
    # dimensionality or type
    # first call fit
    ts.fit(x_train=np.array(x_train), y_train=np.array(y_train))
    # Now predict a new point
    x_test = [[1]]

    with pytest.raises(ValueError):
        ts.predict(x=x_test)  # wrong type

    with pytest.raises(ValueError):
        ts.predict(x=np.array(x_test))  # wrong dimensionality


def test_temperature_scaling_fit_predict():
    x_train = np.array([[1, 1], [2, 3.5]])
    y_train = np.array([[0.9, 0.1], [0.2, 0.8]])

    x_val = np.array([[0, 2]])
    y_val = np.array([[0.8, 0.2]])

    ts = TemperatureCalibration(epochs=2000)

    assert ts.epochs == 2000
    assert ts.temperature == 1.0
    assert len(ts.history) == 0
    assert ts.n_classes is None

    ts.fit(x_train=x_train, y_train=y_train)

    assert ts.temperature != 1.0
    assert len(ts.history) == 2000
    assert ts.n_classes == 2

    ts = TemperatureCalibration(epochs=5000)

    assert ts.epochs == 5000

    # This will cause early stopping
    ts.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    assert ts.temperature > 0.0  # temperature should be positive
    assert len(ts.history) < 5000
    assert ts.n_classes == 2

    # Check that predict returns data of the same dimensionality as
    # the training data.
    x_test = np.array([[0, 1]])
    y_pred = ts.predict(x=x_test)

    assert y_pred.shape == (1, 2)
    assert np.sum(y_pred) == pytest.approx(1.0)
    # Temperature scaling does not change the predicted class
    assert y_pred[0, 1] > y_pred[0, 0]

    #
    # Test for binary classification
    #
    x_train = np.array([1, 3.5])
    y_train = np.array([1, 0])

    ts = TemperatureCalibration()

    #
    ts.fit(x_train=x_train, y_train=y_train)

    assert ts.n_classes == 1

    # Check that predict returns data of the same dimensionality as
    # the training data.
    x_test = np.array([[0.7]])
    y_pred = ts.predict(x=x_test)

    assert y_pred.shape == (1, 1)


#
# Test for class IsotonicCalibration
#
def test_isotonic_calibration_bad_input_type():
    ic = IsotonicCalibration()

    x_train = [[1, 1], [2, 3.5]]
    y_train = [[0.9, 0.1], [0.2, 0.8]]

    with pytest.raises(ValueError):
        ic.fit(x_train=None, y_train=None)

    # Test mixed types
    with pytest.raises(ValueError):
        ic.fit(x_train=None, y_train=np.array(y_train))

    # Test mixed types
    with pytest.raises(ValueError):
        ic.fit(x_train=np.array(x_train), y_train=None)

    with pytest.raises(ValueError):  # y_train should be np.ndarray not list
        ic.fit(x_train=np.array(x_train), y_train=y_train)

    with pytest.raises(ValueError):  # x_train should be np.ndarray not list
        ic.fit(x_train=x_train, y_train=np.array(y_train))

    # some tests for the predict method where the input data have the wrong
    # dimensionality or type
    # first call fit
    ic.fit(x_train=np.array(x_train), y_train=np.array(y_train))
    # Now predict a new point
    x_test = [[1]]

    with pytest.raises(ValueError):
        ic.predict(x=x_test)  # wrong type

    with pytest.raises(ValueError):
        ic.predict(x=np.array(x_test))  # wrong dimensionality


def test_isotonic_calibration_fit_predict():

    # Some tests for the multi-class case
    x_train = np.array([[1, 1], [2, 3.5]])
    y_train = np.array([[0.9, 0.1], [0.2, 0.8]])

    ic = IsotonicCalibration()

    assert len(ic.regressors) == 0

    ic.fit(x_train=x_train, y_train=y_train)

    assert ic.n_classes == 2

    # Check that predict returns data of the same dimensionality as
    # the training data.
    x_test = np.array([[0, 1]])
    y_pred = ic.predict(x=x_test)

    assert y_pred.shape == (1, 2)
    assert np.sum(y_pred) == pytest.approx(1.0)

    # Some tests for the binary classification case.
    x_train = np.array([0.1, 0.5, 0.2])
    y_train = np.array([0.2, 0.55, 0.3])

    ic = IsotonicCalibration()

    assert len(ic.regressors) == 0

    ic.fit(x_train=x_train, y_train=y_train)

    assert ic.n_classes == 1

    # Check that predict returns data of the same dimensionality as
    # the training data.
    x_test = np.array([0.5, 0.1])
    y_pred = ic.predict(x=x_test)

    assert y_pred.shape == (2, 1)


#
# Tests for method expected_calibration_error
#
def test_expected_calibration_error():

    pp = [0.1, 0.5, 0.8, 0.2]
    ac = [0.1, 0.3, 0.5, 0.8, 0.9]
    co = [0.15, 0.3, 0.55, 0.75, 0.92]

    # Test passing invalid parameter values, e.g., type.
    with pytest.raises(ValueError):
        expected_calibration_error(
            prediction_probabilities=pp, accuracy=ac, confidence=co
        )

    with pytest.raises(ValueError):
        expected_calibration_error(
            prediction_probabilities=pp, accuracy=np.array(ac), confidence=np.array(co)
        )

    with pytest.raises(ValueError):
        expected_calibration_error(
            prediction_probabilities=np.array(pp), accuracy=ac, confidence=np.array(co)
        )

    with pytest.raises(ValueError):
        expected_calibration_error(
            prediction_probabilities=np.array(pp), accuracy=np.array(ac), confidence=co
        )

    # test different length for accuracy and confidence parameter
    with pytest.raises(ValueError):
        expected_calibration_error(
            prediction_probabilities=np.array(pp),
            accuracy=np.array([0.2, 0.5]),
            confidence=np.array([0.3, 0.5, 0.8]),
        )

    ece = expected_calibration_error(
        prediction_probabilities=np.array(pp),
        accuracy=np.array(ac),
        confidence=np.array(co),
    )

    assert ece > 0


#
# Tests for method plot_reliability_diagram
#
def test_plot_reliability_diagram():

    cd = ((0.1, 0.5), (0.8, 0.2))
    cd_valid = [(np.array([0.1]), np.array([0.5])), (np.array([0.8]), np.array([0.2]))]
    pr = [0.1, 0.3, 0.5, 0.8, 0.9]
    ece = None
    filename = None

    # Test passing invalid parameter values, e.g., type.
    with pytest.raises(ValueError):
        plot_reliability_diagram(
            calibration_data=cd, predictions=np.array(pr), ece=ece, filename=filename
        )

    with pytest.raises(ValueError):
        plot_reliability_diagram(
            calibration_data=cd_valid, predictions=pr, ece=ece, filename=filename
        )

    with pytest.raises(ValueError):
        plot_reliability_diagram(
            calibration_data=cd_valid,
            predictions=np.array(pr),
            ece=0.5,  # should be list of floats or None
            filename=filename,
        )

    with pytest.raises(ValueError):
        plot_reliability_diagram(
            calibration_data=cd_valid, predictions=np.array(pr), ece=ece, filename=10
        )  # should be string or None
