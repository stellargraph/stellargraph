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
Ensembles of graph neural network models, GraphSAGE, GCN, GAT.
"""

__all__ = ["Ensemble"]

import numpy as np
import keras as K


class Ensemble(object):
    """

    """

    def __init__(self, model, n_estimators=3, n_predictions=3):
        """

        Args:
            model: A keras model.
            n_estimators: The number of estimators/models in the ensemble
            n_predictions: The number of predictions per query point per estimator
        """
        self.models = []
        self.history = []
        self.n_estimators = n_estimators
        self.n_predictions = n_predictions

        self._init_models(model)

    def _init_models(self, model):

        # first copy is the given model
        self.models.append(model)
        # now clone the model self.n_estimators-1 times
        for _ in range(self.n_estimators - 1):
            self.models.append(K.models.clone_model(model))

    def compile(
        self,
        optimizer,
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
    ):
        for model in self.models:
            model.compile(
                optimizer,
                loss,
                metrics,
                loss_weights,
                sample_weight_mode,
                weighted_metrics,
            )

    def fit_generator(
        self,
        generator,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        validation_data=None,
        validation_steps=None,
        validation_freq=1,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    ):
        self.history = []
        for model in self.models:
            self.history.append(
                model.fit_generator(
                    generator,
                    steps_per_epoch,
                    epochs,
                    verbose,
                    validation_data,
                    validation_steps,
                    validation_freq,
                    class_weight,
                    max_queue_size,
                    workers,
                    use_multiprocessing,
                    shuffle,
                    initial_epoch,
                )
            )

    def evaluate_generator(
        self,
        generator,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        train_metrics = []
        for model in self.models:
            tm = []
            for _ in range(self.n_predictions):
                tm.append(
                    model.evaluate_generator(
                        generator,
                        steps,
                        max_queue_size,
                        workers,
                        use_multiprocessing,
                        verbose,
                    )
                )
            train_metrics.append(np.mean(tm, axis=0))
        return np.mean(train_metrics, axis=0), np.std(train_metrics, axis=0)

    def predict_generator(
        self,
        generator,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        predictions = []

        for model in self.models:
            model_predictions = []
            for _ in range(self.n_predictions):
                model_predictions.append(
                    model.predict_generator(
                        generator,
                        steps,
                        max_queue_size,
                        workers,
                        use_multiprocessing,
                        verbose,
                    )
                )
            # average the predictions
            model_predictions = np.mean(model_predictions, axis=0)

            # add to predictions list
            predictions.append(predictions)

        return predictions

    def metrics_names(self):
        return self.models[
            0
        ].metrics_names  # assumes all models are same as it should be
