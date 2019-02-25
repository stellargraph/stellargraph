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

from stellargraph.layer import *

__all__ = ["Ensemble"]

import numpy as np
import keras as K

import stellargraph as sg


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

    def layers(self, indx=None):
        """

        Args:
            indx:

        Returns:

        """
        if indx is None and len(self.models) > 0:
            # Default is to return the layers for the first model
            return self.models[0].layers

        if len(self.models) > indx:
            return self.models[indx].layers
        else:
            # Error because index is out of bounds
            raise ValueError(
                "indx {} is out of range 0 to {}".format(indx, len(self.models))
            )

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
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                sample_weight_mode=sample_weight_mode,
                weighted_metrics=weighted_metrics,
            )

    def fit_generator(
        self,
        generator,
        train_data=None,
        train_targets=None,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        validation_data=None,
        validation_steps=None,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    ):

        if train_data is not None and not isinstance(
            generator,
            (
                sg.mapper.node_mappers.GraphSAGENodeGenerator,
                sg.mapper.node_mappers.HinSAGENodeGenerator,
                sg.mapper.node_mappers.FullBatchNodeGenerator,
            ),
        ):
            raise ValueError(
                "generator must be of type GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator if you want to use Bagging. Received type {}".format(
                    type(generator)
                )
            )

        self.history = []

        if train_data is not None:
            # Prepare the training data for each model. Use sampling with replacement to create len(self.models)
            # datasets.
            print("*** Train with Bagging ***")
            for model in self.models:
                di_index = np.random.choice(
                    len(train_data), size=len(train_data)
                )  # sample with replacement
                di_train = train_data[di_index]
                di_targets = None
                if train_targets is not None:
                    di_targets = train_targets[di_index]

                print(
                    "Unique train data {} and targets {}".format(
                        len(np.unique(di_train)), len(np.unique(di_targets))
                    )
                )

                di_gen = generator.flow(di_train, di_targets)
                self.history.append(
                    model.fit_generator(
                        generator=di_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        class_weight=class_weight,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        shuffle=shuffle,
                        initial_epoch=initial_epoch,
                    )
                )
        else:
            for model in self.models:
                self.history.append(
                    model.fit_generator(
                        generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        class_weight=class_weight,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        shuffle=shuffle,
                        initial_epoch=initial_epoch,
                    )
                )

        return self.history

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
                        generator=generator,
                        steps=steps,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        verbose=verbose,
                    )
                )
            train_metrics.append(np.mean(tm, axis=0))
        return np.mean(train_metrics, axis=0), np.std(train_metrics, axis=0)

    def predict_generator(
        self,
        generator,
        summarise=None,
        output_layer=None,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        predictions = []

        if output_layer is not None:
            predict_models = [
                K.Model(inputs=model.input, outputs=model.layers[output_layer].output)
                for model in self.models
            ]
        else:
            predict_models = self.models

        for model in predict_models:
            model_predictions = []
            for _ in range(self.n_predictions):
                model_predictions.append(
                    model.predict_generator(
                        generator=generator,
                        steps=steps,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        verbose=verbose,
                    )
                )
            # average the predictions
            if summarise is not None:
                model_predictions = np.mean(model_predictions, axis=0)

            # add to predictions list
            predictions.append(model_predictions)

        predictions = np.array(predictions)
        if len(predictions.shape) > 4:
            predictions = predictions.reshape(predictions.shape[0:3] + (-1,))

        return predictions

    def metrics_names(self):
        return self.models[
            0
        ].metrics_names  # assumes all models are same as it should be
