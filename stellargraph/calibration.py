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

from sklearn.linear_model import LogisticRegression


def expected_calibration_error(prediction_probabilities, accuracy, confidence):
    """
    Helper function for calculating the expected calibration error as defined in
    the paper On Calibration of Modern Neural Networks, C. Guo, et. al., ICML, 2017

    It is assumed that for a validation dataset, the prediction probabilities have
    been calculated for each point in the dataset and given in the array
    prediction_probabilities.

    Args:
        prediction_probabilities (numpy array):  The predicted probabilities.
        accuracy (numpy array): The accuracy such that the i-th entry in the array holds the proportion of correctly
            classified samples that fall in the i-th bin.
        confidence (numpy array): The confidence such that the i-th entry in the array is the average prediction
            probability over all the samples assigned to this bin.

    Returns:
        float: The expected calibration error.

    """
    if not isinstance(prediction_probabilities, np.ndarray):
        raise ValueError(
            "Parameter prediction_probabilities must be type numpy.ndarray but given object of type {}".format(
                type(prediction_probabilities).__name__
            )
        )
    if not isinstance(accuracy, np.ndarray):
        raise ValueError(
            "Parameter accuracy must be type numpy.ndarray but given object of type {}".format(
                type(accuracy).__name__
            )
        )
    if not isinstance(confidence, np.ndarray):
        raise ValueError(
            "Parameter confidence must be type numpy.ndarray but given object of type {}".format(
                type(confidence).__name__
            )
        )

    if len(accuracy) != len(confidence):
        raise ValueError(
            "Arrays accuracy and confidence should have the same size but instead received {} and {} respectively.".format(
                len(accuracy), len(confidence)
            )
        )

    n_bins = len(accuracy)  # the number of bins
    n = len(prediction_probabilities)  # number of samples
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
        calibration_data (list): The calibration data as a list where each entry in the list is a 2-tuple of type
            numpy.ndarray. Each entry in the tuple holds the fraction of positives and the mean predicted values
            for the true and predicted class labels.
        predictions (np.ndarray): The probabilistic predictions of the classifier for each sample in the dataset used
            for diagnosing miscalibration.
        ece (None or list of float): If not None, this list stores the expected calibration error for each class.
        filename (str or None): If not None, the figure is saved on disk in the given filename.
    """
    if not isinstance(calibration_data, list):
        raise ValueError(
            "Parameter calibration_data should be list of 2-tuples but received type {}".format(
                type(calibration_data).__name__
            )
        )

    if not isinstance(predictions, np.ndarray):
        raise ValueError(
            "Parameter predictions should be of type numpy.ndarray but received type {}".format(
                type(predictions).__name__
            )
        )
    if ece is not None and not isinstance(ece, list):
        raise ValueError(
            "Parameter ece should be None or list of floating point numbers but received type {}".format(
                type(ece).__name__
            )
        )
    if filename is not None and not isinstance(filename, str):
        raise ValueError(
            "Parameter filename should be None or str type but received type {}".format(
                type(filename).__name__
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
    A class for temperature calibration for binary and multi-class classification problems.

    For binary classification, Platt Scaling is used for calibration. Platt Scaling was
    proposed in the paper Probabilistic outputs for support vector machines and comparisons to regularized
    likelihood methods, J. C. Platt, Advances in large margin classifiers, 10(3): 61-74, 1999.

    For multi-class classification, Temperature Calibration is used. It is an extension of Platt Scaling
    and it was proposed in the paper On Calibration of Modern Neural Networks, C. Guo et. al., ICML, 2017.

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
        self.early_stopping = False
        self.lr = None  # The logistic regression model for Platt scaling

    def _fit_temperature_scaling(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the calibration model using Temperature Scaling.

        If validation data is given, then training stops when the validation accuracy starts increasing.

        Args:
            x_train (numpy array): The training data that should be a classifier's non-probabilistic outputs. It should
                have shape (N, C) where N is the number of samples and C is the number of classes.
            y_train (numpy array): The training data class labels. It should have shape (N, C) where N is the number
                of samples and C is the number of classes and the class labels are one-hot encoded.
            x_val (numpy array or None): The validation data used for early stopping. It should have shape (M, C) where
                M is the number of validation samples and C is the number of classes and the class labels are one-hot
                encoded.
            y_val (numpy array or None): The validation data class labels. It should have shape (M, C) where M is the
                number of validation samples and C is the number of classes and the class labels are one-hot encoded.
        """

        T = tf.Variable(tf.ones(shape=(1,)), name="T")

        def cost(T, x, y):

            scaled_logits = tf.multiply(name="z", x=x, y=1.0 / T)

            cost_value = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits, labels=y)
            )

            return cost_value

        def grad(T, x, y):

            with tf.GradientTape() as tape:
                cost_value = cost(T, x, y)

            return cost_value, tape.gradient(cost_value, T)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        for epoch in range(self.epochs):
            train_cost, grads = grad(T, x_train, y_train)
            optimizer.apply_gradients(zip([grads], [T]))
            if self.early_stopping:
                val_cost = cost(T, x_val, y_val)
                if (len(self.history) > 0) and (val_cost > self.history[-1][1]):
                    break
                else:  # keep going
                    self.history.append([train_cost, val_cost, T.numpy()[0]])
            else:
                self.history.append([train_cost, T.numpy()[0]])

        self.history = np.array(self.history)
        self.temperature = self.history[-1, -1]

    def _fit_platt_scaling(self, x_train, y_train):
        """
        Helper method for calibration of a binary classifier using Platt Scaling.

        Args:
            x_train (numpy array): The training data that should be a classifier's non-probabilistic outputs. It
                should have shape (N,) where N is the number of training samples.
            y_train (numpy array): The training data class labels. It should have shape (N,) where N is the number
                of training samples.

        """

        self.lr = LogisticRegression(fit_intercept=True, verbose=False)

        self.lr.fit(x_train, y_train)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the calibration model.

        For temperature scaling of a multi-class classifier, If validation data is given, then
        training stops when the validation accuracy starts increasing. Validation data are ignored for Platt scaling

        Args:
            x_train (numpy array): The training data that should be a classifier's non-probabilistic outputs. For
                calibrating a binary classifier it should have shape (N,) where N is the number of training samples.
                For calibrating a multi-class classifier, it should have shape (N, C) where N is the number of samples
                and C is the number of classes.
            y_train (numpy array): The training data class labels. For
                calibrating a binary classifier it should have shape (N,) where N is the number of training samples.
                For calibrating a multi-class classifier, it should have shape (N, C) where N is the number of samples
                and C is the number of classes and the class labels are one-hot encoded.
            x_val (numpy array or None): The validation data used only for calibrating multi-class classification
                models. It should have shape (M, C) where M is the number of validation samples and C is the number of
                classes and the class labels are one-hot encoded.
                that should be the classifier's non-probabilistic outputs.
            y_val (numpy array or None): The validation data class labels used only for calibrating multi-class
                classification models. It should have shape (M, C) where M is the number of validation samples and C
                is the number of classes and the class labels are one-hot encoded.
        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("x_train and y_train must be numpy arrays")

        if (x_val is not None and y_val is None) or (
            x_val is None and y_val is not None
        ):
            raise ValueError(
                "Either both x_val and y_val should be None or both should be numpy arrays."
            )

        if x_val is not None and y_val is not None:
            if not isinstance(x_val, np.ndarray) or not isinstance(y_val, np.ndarray):
                raise ValueError("x_train and y_train must be numpy arrays")

            self.early_stopping = True
            print(
                "Using Early Stopping based on performance evaluated on given validation set."
            )

        if len(x_train.shape) == 1:
            self.n_classes = 1
        else:
            self.n_classes = x_train.shape[1]

        if self.n_classes > 1:
            self._fit_temperature_scaling(x_train, y_train, x_val, y_val)
        else:
            self._fit_platt_scaling(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

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

    def predict(self, x):
        """
        This method calibrates the given data using the learned temperature. It
        scales each logit by the temperature, exponentiates the results, and finally
        normalizes the scaled values such that their sum is 1.

        Args:
            x (numpy.ndarray): The logits. For binary classification problems, it should have dimensionality (N,) where
                N is the number of samples to calibrate. For multi-class problems, it should have dimensionality (N, C)
                where C is the number of classes.

        Returns:
            numpy array: The calibrated probabilities.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "x should be numpy.ndarray but received {}".format(type(x).__name__)
            )

        if len(x.shape) > 1 and x.shape[1] != self.n_classes:
            raise ValueError(
                "Expecting input vector of dimensionality {} but received {}".format(
                    self.n_classes, len(x)
                )
            )
        x_ = x

        if self.n_classes == 1:
            return self.lr.predict_proba(X=x)[:, 1].reshape(-1, 1)
        else:
            scaled_prediction = x_ / self.temperature

            return np.exp(scaled_prediction) / np.sum(
                np.exp(scaled_prediction), axis=-1, keepdims=True
            )


class IsotonicCalibration(object):
    """
    A class for applying Isotonic Calibration to the outputs of a binary or multi-class classifier.
    """

    def __init__(self):
        self.n_classes = None
        self.regressors = []

    def fit(self, x_train, y_train):
        """
        Train a calibration model using the provided data.

        Args:
            x_train (numpy array): The training data that should be the classifier's probabilistic outputs. It should
                have shape NxC where N is the number of training samples and C is the number of classes.
            y_train (numpy array): The training class labels. For binary problems y_train has shape (N,)
                when N is the number of samples. For multi-class classification, y_train has shape (N,C) where
                C is the number of classes and y_train is using one-hot encoding.

        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError(
                "x_train and y_train should be type numpy.ndarray but received {} and {}".format(
                    type(x_train).__name__, type(y_train).__name__
                )
            )

        if len(x_train.shape) == 1:
            self.n_classes = 1
        else:
            self.n_classes = x_train.shape[1]

        if self.n_classes == 1:
            self.regressors.append(IsotonicRegression(out_of_bounds="clip"))
            if len(x_train.shape) > 1:
                x_train = x_train.reshape(-1)
            self.regressors[-1].fit(X=x_train.astype(np.double), y=y_train)
        else:
            for n in range(self.n_classes):
                self.regressors.append(IsotonicRegression(out_of_bounds="clip"))
                self.regressors[-1].fit(
                    X=x_train[:, n].astype(np.double), y=y_train[:, n]
                )

    def predict(self, x):
        """
        This method calibrates the given data assumed the output of a classification model.

        For multi-class classification, the probabilities for each class are first scaled using the corresponding
        isotonic regression model and then normalized to sum to 1.

        Args:
            x (numpy array): The values to calibrate. For binary classification problems it should have shape (N,) where
                N is the number of samples to calibrate. For multi-class classification problems, it should have shape
                (N, C) where C is the number of classes.

        Returns:
            numpy array: The calibrated probabilities. It has shape (N, C) where N is the number of samples
            and C is the number of classes.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "x should be numpy.ndarray but received {}".format(type(x).__name__)
            )

        if self.n_classes > 1 and x.shape[1] != self.n_classes:
            raise ValueError(
                "Expecting input vector of dimensionality {} but received {}".format(
                    self.n_classes, len(x)
                )
            )

        if self.n_classes == 1:
            x = x.reshape(-1, 1)

        predictions = []
        for n in range(self.n_classes):
            predictions.append(self.regressors[n].transform(T=x[:, n]))

        predictions = np.transpose(np.array(predictions))

        if self.n_classes > 1:
            predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

        return predictions
