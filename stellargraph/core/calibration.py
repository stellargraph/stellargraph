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
Calibration for classification, binary and multi-class, models.
"""

__all__ = [
    "IsotonicCalibration",
    "TemperatureCalibration",
    "expected_calibration_error",
    "plot_reliability_diagram",
]

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(prediction_probabilities, accuracy, confidence):
    """
    Helper function for calculating the expected calibration error as defined in
    the paper On Calibration of Modern Neural Networks, C. Guo, et. al., ICML, 2017

    Args:
        prediction_probabilities: <numpy array>  The predicted probabilities
        accuracy: <numpy array> The accuracy
        confidence: <numpy array> The confidence

    Returns: <Float> The expected calibration error.

    """
    if not isinstance(prediction_probabilities, np.ndarray):
        raise ValueError(
            "Parameter prediction_probabilities must be type numpy.ndarray but given object of type {}".format(
                type(prediction_probabilities)
            )
        )
    if not isinstance(accuracy, np.ndarray):
        raise ValueError(
            "Parameter accuracy must be type numpy.ndarray but given object of type {}".format(
                type(accuracy)
            )
        )
    if not isinstance(confidence, np.ndarray):
        raise ValueError(
            "Parameter confidence must be type numpy.ndarray but given object of type {}".format(
                type(confidence)
            )
        )

    if len(accuracy) != len(confidence):
        raise ValueError(
            "Arrays accuracy and confidence should have the same size but instead received {} and {} respectively.".format(
                len(accuracy), len(confidence)
            )
        )

    n_bins = len(accuracy)  # the number of bins
    n = len(prediction_probabilities)  # number of points
    h = np.histogram(a=prediction_probabilities, range=(0, 1), bins=n_bins)[
        0
    ]  # just the counts
    ece = 0
    for m in np.arange(n_bins):
        ece = ece + (h[m] / n) * np.abs(accuracy[m] - confidence[m])
    return ece


def plot_reliability_diagram(calibration_data, predictions, ece=None, filename=None):
    """
    Helper function for plotting a reliability diagram.
    Args:
        calibration_data: <list> The calibration data as list where each entry in the
        list is a 2-tuple of type numpy.ndarray. Each entry in the tuple holds the
        fraction of positives and the mean predicted values for the true and predicted
        class labels.
        predictions: <np.ndarray> The probabilistic predictions for the data used in
        diagnosing miscalibration.
        ece: <None or list of floats> The expected calibration error for each class
        filename: <string or None> If not None, the figure is saved on disk in the given filename.
    """
    if not isinstance(calibration_data, list):
        raise ValueError(
            "Parameter calibration_data should be list of 2-tuples but received type {}".format(
                type(calibration_data)
            )
        )
    if not isinstance(predictions, np.ndarray):
        raise ValueError(
            "Parameter predictions should be of type numpy.ndarray but received type {}".format(
                type(predictions)
            )
        )
    if ece is not None and not isinstance(ece, list):
        raise ValueError(
            "Parameter ece should be None or list of floating point numbers but received type {}".format(
                type(ece)
            )
        )
    if filename is not None and not isinstance(filename, str):
        raise ValueError(
            "Parameter filename should be None or str type but received type {}".format(
                type(filename)
            )
        )

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((6, 1), (4, 0))

    if ece is not None:
        calibration_error = ",".join(format(e, " 0.4f") for e in ece)

    for i, data in enumerate(calibration_data):
        fraction_of_positives, mean_predicted_value = data
        # print(fraction_of_positives, mean_predicted_value)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", alpha=1.0)
        if ece is not None:
            ax1.set_title("Calibration Curve (ECE={})".format(calibration_error))
        ax1.set_xlabel("Mean Predicted Value", fontsize=16)
        ax1.set_ylabel("Fraction of Positives", fontsize=16)
        ax1.plot([0, 1], [0, 1], "g--")
        ax2.hist(predictions[:, i], range=(0, 1), bins=10, histtype="step", lw=2)
        ax2.set_xlabel("Bin", fontsize=16)
        ax2.set_ylabel("Count", fontsize=16)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")


class TemperatureCalibration(object):
    """
    A class for temperature calibration for multi-class classification problems. Temperature
    Calibration is an extension of Platt Scaling and it was proposed in the paper
    On Calibration of Modern Neural Networks, C. Guo et. al., ICML, 2017.

    In Temperature Calibration, a classifier's non-probabilistic outputs, i.e., logits, are
    scaled by a trainable parameter called Temperature. The softmax is applied to the rescaled
    logits to calculate the probabilistic output. As noted in the cited paper, Temperature
    Scaling does not change the maximum of the softmax function so the classifier's prediction
    remain the same.
    """

    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.n_classes = None
        self.temperature = 1.0  # default is no scaling
        self.history = []

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the model. If validation data is given, then training stops when the
        validation accuracy starts increasing. This prevent overfitting.
        Args:
            x_train: <numpy array> The training data that should be the classifier's non-probabilistic outputs.
            y_train: <numpy array> The training data class labels as one hot encoded vectors
            x_val: <numpy array or None> The validation data that should be the classifier's non-probabilistic outputs.
            y_val: <numpy array or None> The validation data class labels as one hot encoded vectors

        Returns: <float> The estimate temperature

        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("x_train and y_train must be numpy arrays")

        early_stopping = False
        if (x_val is not None and y_val is None) or (
            x_val is None and y_val is not None
        ):
            raise ValueError(
                "Either both x_val and y_val should be None or both should be numpy arrays."
            )

        if x_val is not None and y_val is not None:
            if not isinstance(x_val, np.ndarray) or not isinstance(y_val, np.ndarray):
                raise ValueError("x_train and y_train must be numpy arrays")

            early_stopping = True
            print(
                "Using Early Stopping based on performance evaluated on given validation set."
            )

        self.n_classes = x_train.shape[1]
        # Specify the tensorflow program.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            x = tf.placeholder(
                tf.float32, [None, self.n_classes], name="x"
            )  # input are the model logits
            y = tf.placeholder(
                tf.float32, [None, self.n_classes], name="y"
            )  # output is one-hot encoded true class labels

            T = tf.get_variable(
                "T", [1], initializer=tf.ones_initializer
            )  # the temperature

            scaled_logits = tf.multiply(
                name="z", x=x, y=1.0 / T
            )  # logits scaled by inverse T

            # cost function to optimise
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=scaled_logits, labels=y
                )
            )

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        self.history = []
        for epoch in range(self.epochs):
            _, c, t = sess.run([optimizer, cost, T], feed_dict={x: x_train, y: y_train})
            if early_stopping:
                c_val = sess.run([cost], feed_dict={x: x_val, y: y_val})
                if len(self.history) > 10 and c_val > self.history[-1][1]:
                    break
                else:  # keep going
                    self.history.append([c, c_val[0], t[0]])
            else:
                self.history.append([c, t[0]])
        self.history = np.array(self.history)
        self.temperature = self.history[-1, -1]

        return self.temperature

    def plot_training_history(self):
        """
        Helper function for plotting the training history.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 5))
        ax1.plot(self.history[:, 0], label="Training")
        if self.history.shape[1] == 3:  # has validation cost
            ax1.plot(self.history[:, 1], label="Validation")
        ax1.set_title("Cost")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cost")
        ax1.legend(loc="upper right")
        ax2.plot(self.history[:, -1])
        ax2.set_title("Temperature")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Temperature")

    def transform(self, x):
        """
        This method calibrates the classifier's output using the learned temperature. It
        scales each logit by the temperature, exponentiates the results, and finally
        normalizes the scaled values such that their sum is 1.
        Args:
            x: <numpy.ndarray> The classifier's non-probabilistic outputs to calibrate.

        Returns: The calibrated probabilities.

        """
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "x should be numpy.ndarray but received {}".format(type(x))
            )

        if x.shape[1] != self.n_classes:
            raise ValueError(
                "Expecting input vector of dimensionality {} but received {}".format(
                    self.n_classes, len(x)
                )
            )

        scaled_prediction = x / self.temperature
        return np.exp(scaled_prediction) / np.sum(
            np.exp(scaled_prediction), axis=-1, keepdims=True
        )


class IsotonicCalibration(object):
    """
    A class for applying Isotonic Calibration to the outputs of a multi-class classifier.
    """

    def __init__(self):
        self.n_classes = None
        self.regressors = []

    def fit(self, x_train, y_train):
        """
        Train the model.
        Args:
            x_train: The training data that should be the classifier's probabilistic outputs
            y_train: The training class labels as one hot encoded vectors.

        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError(
                "x_train and y_train should be type numpy.ndarray but received {} and {}".format(
                    type(x_train), type(y_train)
                )
            )

        self.n_classes = x_train.shape[1]

        for n in range(self.n_classes):
            self.regressors.append(IsotonicRegression(out_of_bounds="clip"))

            self.regressors[-1].fit(X=x_train[:, n], y=y_train[:, n])

    def transform(self, x):
        """
        This method calibrates the classifier's output using the trained regressors. The
        probabilities for each class are first scaled using the corresponding regression model
        and then normalized to sum to 1.
        Args:
            x: <numpy array> The classifier's probabilistic outputs to calibrate.

        Returns: <numpy array> The calibrated probabilities.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "x should be numpy.ndarray but received {}".format(type(x))
            )

        if x.shape[1] != self.n_classes:
            raise ValueError(
                "Expecting input vector of dimensionality {} but received {}".format(
                    self.n_classes, len(x)
                )
            )

        predictions = []
        for n in range(self.n_classes):
            predictions.append(self.regressors[n].transform(T=x[:, n]))

        predictions = np.transpose(np.array(predictions))
        predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

        return predictions
