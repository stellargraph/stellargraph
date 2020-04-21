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
Ensembles of graph neural network models, GraphSAGE, GCN, GAT, and HinSAGE, with optional bootstrap sampling of the
training data (implemented in the BaggingEnsemble class).
"""

from stellargraph.layer import *

__all__ = ["Ensemble", "BaggingEnsemble"]

import numpy as np
from tensorflow import keras as K
from tensorflow.keras.callbacks import EarlyStopping

import stellargraph as sg


class Ensemble(object):
    """
    The Ensemble class can be used to create ensembles of stellargraph's graph neural network algorithms including
    GCN, GraphSAGE, GAT, and HinSAGE. Ensembles can be used for training classification and regression problems for
    node attribute inference and link prediction.

    The Ensemble class can be used to create Naive ensembles.

    Naive ensembles add model diversity by random initialisation of the models' weights (before training) to
    different values. Each model in the ensemble is trained on the same training set of examples.

    """

    def __init__(self, model, n_estimators=3, n_predictions=3):
        """

        Args:
            model: A keras model.
            n_estimators (int):  The number of estimators (aka models) in the ensemble.
            n_predictions (int):  The number of predictions per query point per estimator
        """
        if not isinstance(model, K.Model):
            raise ValueError(
                "({}) model must be a Keras model received object of type {}".format(
                    type(self).__name__, type(model).__name__
                )
            )
        if n_estimators <= 0 or not isinstance(n_estimators, int):
            raise ValueError(
                "({}) n_estimators must be positive integer but received {}".format(
                    type(self).__name__, n_estimators
                )
            )

        if n_predictions <= 0 or not isinstance(n_predictions, int):
            raise ValueError(
                "({}) n_predictions must be positive integer but received {}".format(
                    type(self).__name__, n_predictions
                )
            )

        self.metrics_names = (
            None  # It will be set when the self.compile() method is called
        )
        self.models = []
        self.history = []
        self.n_estimators = n_estimators
        self.n_predictions = n_predictions
        self.early_stoppping_patience = 10

        # Create the enseble from the given base model
        self._init_models(model)

    def _init_models(self, model):
        """
        This method creates an ensemble of models by cloning the given base model self.n_estimators times.

        All models have the same architecture but their weights are initialised with different (random) values.

        Args:
            model: A Keras model that is the base model for the ensemble.

        """
        # first copy is the given model
        self.models.append(model)
        # now clone the model self.n_estimators-1 times
        for _ in range(self.n_estimators - 1):
            self.models.append(K.models.clone_model(model))

    def layers(self, indx=None):
        """
        This method returns the layer objects for the model specified by the value of indx.

        Args:
            indx (None or int): The index  (starting at 0) of the model to return the layers for.
                If it is None, then the layers for the 0-th (or first) model are returned.

        Returns:
            list: The layers for the specified model.

        """
        if indx is not None and not isinstance(indx, (int,)):
            raise ValueError(
                "({}) indx should be None or integer type but received type {}".format(
                    type(self).__name__, type(indx).__name__
                )
            )
        if isinstance(indx, (int,)) and indx < 0:
            raise ValueError(
                "({}) indx must be greater than or equal to zero but received {}".format(
                    type(self).__name__, indx
                )
            )

        if indx is None and len(self.models) > 0:
            # Default is to return the layers for the first model
            return self.models[0].layers

        if len(self.models) > indx:
            return self.models[indx].layers
        else:
            # Error because index is out of bounds
            raise ValueError(
                "({}) indx {} is out of range 0 to {}".format(
                    type(self).__name__, indx, len(self.models)
                )
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
        """
        Method for configuring the model for training. It is a wrapper of the `keras.models.Model.compile` method for
        all models in the ensemble.

        For detailed descriptions of Keras-specific parameters consult the Keras documentation
        at https://keras.io/models/sequential/

        Args:
            optimizer (Keras optimizer or str): (Keras-specific parameter) The optimizer to use given either as an
                instance of a keras optimizer or a string naming the optimiser of choice.
            loss (Keras function or str): (Keras-specific parameter) The loss function or string indicating the
                type of loss to use.
            metrics (list or dict): (Keras-specific parameter) List of metrics to be evaluated by each model in
                the ensemble during training and testing. It should be a list for a model with a single output. To
                specify different metrics for different outputs of a multi-output model, you could also pass a
                dictionary.
            loss_weights (None or list): (Keras-specific parameter) Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value
                that will be minimized by the model will then be the weighted sum of all individual losses, weighted by
                the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs.
                If a tensor, it is expected to map output names (strings) to scalar coefficients.
            sample_weight_mode (None, str, list, or dict): (Keras-specific parameter) If you need to do
                timestep-wise sample weighting (2D weights), set this to "temporal".  None defaults to sample-wise
                weights (1D). If the model has multiple outputs, you can use a different  sample_weight_mode on
                each output by passing a dictionary or a list of modes.
            weighted_metrics (list): (Keras-specific parameter) List of metrics to be evaluated and weighted by
                sample_weight or class_weight during training and testing.

        """
        for model in self.models:
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                sample_weight_mode=sample_weight_mode,
                weighted_metrics=weighted_metrics,
            )

        self.metrics_names = self.models[0].metrics_names  # assumes all models are same

    def fit(
        self,
        generator,
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
        use_early_stopping=False,
        early_stopping_monitor="val_loss",
    ):
        """
        This method trains the ensemble on the data specified by the generator. If validation data are given, then the
        training metrics are evaluated on these data and results printed on screen if verbose level is greater than 0.

        The method trains each model in the ensemble in series for the number of epochs specified. Training can
        also stop early with the best model as evaluated on the validation data, if use_early_stopping is set to True.

        For detail descriptions of Keras-specific parameters consult the Keras documentation
        at https://keras.io/models/sequential/

        Args:
            generator: The generator object for training data. It should be one of type
                NodeSequence, LinkSequence, SparseFullBatchSequence, or FullBatchSequence.
            steps_per_epoch (None or int): (Keras-specific parameter) If not None, it specifies the number of steps
                to yield from the generator before declaring one epoch finished and starting a new epoch.
            epochs (int): (Keras-specific parameter) The number of training epochs.
            verbose (int): (Keras-specific parameter) The verbocity mode that should be 0 , 1, or 2 meaning silent,
                progress bar, and one line per epoch respectively.
            validation_data: A generator for validation data that is optional (None). If not None then, it should
                be one of type NodeSequence, LinkSequence, SparseFullBatchSequence, or FullBatchSequence.
            validation_steps (None or int): (Keras-specific parameter) If validation_generator is not None, then it
                specifies the number of steps to yield from the generator before stopping at the end of every epoch.
            class_weight (None or dict): (Keras-specific parameter) If not None, it should be a dictionary
                mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during
                training only). This can be useful to tell the model to "pay more attention" to samples from an
                under-represented class.
            max_queue_size (int): (Keras-specific parameter) The maximum size for the generator queue.
            workers (int): (Keras-specific parameter) The maximum number of workers to use.
            use_multiprocessing (bool): (Keras-specific parameter) If True then use process based threading.
            shuffle (bool): (Keras-specific parameter) If True, then it shuffles the order of batches at the
                beginning of each training epoch.
            initial_epoch (int): (Keras-specific parameter) Epoch at which to start training (useful for resuming a
                previous training run).
            use_early_stopping (bool): If set to True, then early stopping is used when training each model
                in the ensemble. The default is False.
            early_stopping_monitor (str): The quantity to monitor for early stopping, e.g., 'val_loss',
                'val_weighted_acc'. It should be a valid Keras metric.

        Returns:
            list: It returns a list of Keras History objects each corresponding to one trained model in the ensemble.

        """
        if not isinstance(
            generator,
            (
                sg.mapper.NodeSequence,
                sg.mapper.LinkSequence,
                sg.mapper.FullBatchSequence,
                sg.mapper.SparseFullBatchSequence,
            ),
        ):
            raise ValueError(
                "({}) If train_data is None, generator must be one of type NodeSequence, LinkSequence, FullBatchSequence "
                "but received object of type {}".format(
                    type(self).__name__, type(generator).__name__
                )
            )

        self.history = []

        es_callback = None
        if use_early_stopping and validation_data is not None:
            es_callback = [
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    patience=self.early_stoppping_patience,
                    restore_best_weights=True,
                )
            ]

        for model in self.models:
            self.history.append(
                model.fit(
                    generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=es_callback,
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

    def fit_generator(self, *args, **kwargs):
        """
        Deprecated: use :meth:`fit`.
        """
        warnings.warn(
            "'fit_generator' has been replaced by 'fit', to match tensorflow.keras.Model",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fit(*args, **kwargs)

    def evaluate(
        self,
        generator,
        test_data=None,
        test_targets=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        """
        Evaluates the ensemble on a data (node or link) generator. It makes `n_predictions` for each data point for each
        of the `n_estimators` and returns the mean and standard deviation of the predictions.

        For detailed descriptions of Keras-specific parameters consult the Keras documentation
        at https://keras.io/models/sequential/

        Args:
            generator: The generator object that, if test_data is not None, should be one of type
                GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator, GraphSAGELinkGenerator,
                or HinSAGELinkGenerator. However, if test_data is None, then generator should be one of type
                NodeSequence, LinkSequence, or FullBatchSequence.
            test_data (None or iterable): If not None, then it is an iterable, e.g. list, that specifies the node IDs
                to evaluate the model on.
            test_targets (None or iterable): If not None, then it is an iterable, e.g. list, that specifies the target
                values for the test_data.
            max_queue_size (int): (Keras-specific parameter) The maximum size for the generator queue.
            workers (int): (Keras-specific parameter) The maximum number of workers to use.
            use_multiprocessing (bool): (Keras-specific parameter) If True then use process based threading.
            verbose (int): (Keras-specific parameter) The verbocity mode that should be 0 or 1 with the former turning
                verbocity off and the latter on.

        Returns:
            tuple: The mean and standard deviation of the model metrics for the given data.

        """
        if test_data is not None and not isinstance(
            generator,
            (
                sg.mapper.GraphSAGENodeGenerator,
                sg.mapper.HinSAGENodeGenerator,
                sg.mapper.FullBatchNodeGenerator,
                sg.mapper.GraphSAGELinkGenerator,
                sg.mapper.HinSAGELinkGenerator,
            ),
        ):
            raise ValueError(
                "({}) generator parameter must be of type GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator, "
                "GraphSAGELinkGenerator, or HinSAGELinkGenerator. Received type {}".format(
                    type(self).__name__, type(generator).__name__
                )
            )
        elif not isinstance(
            generator,
            (
                sg.mapper.NodeSequence,
                sg.mapper.LinkSequence,
                sg.mapper.FullBatchSequence,
                sg.mapper.SparseFullBatchSequence,
            ),
        ):
            raise ValueError(
                "({}) If test_data is None, generator must be one of type NodeSequence, "
                "LinkSequence, FullBatchSequence, or SparseFullBatchSequence "
                "but received object of type {}".format(
                    type(self).__name__, type(generator).__name__
                )
            )
        if test_data is not None and test_targets is None:
            raise ValueError("({}) test_targets not given.".format(type(self).__name__))

        data_generator = generator
        if test_data is not None:
            data_generator = generator.flow(test_data, test_targets)

        test_metrics = []
        for model in self.models:
            tm = []
            for _ in range(self.n_predictions):
                tm.append(
                    model.evaluate(
                        data_generator,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        verbose=verbose,
                    )  # Keras evaluate_generator returns a scalar
                )
            test_metrics.append(np.mean(tm, axis=0))

        # Return the mean and standard deviation of the metrics
        return np.mean(test_metrics, axis=0), np.std(test_metrics, axis=0)

    def evaluate_generator(self, *args, **kwargs):
        """
        Deprecated: use :meth:`evaluate`.
        """
        warnings.warn(
            "'evaluate_generator' has been replaced by 'evaluate', to match tensorflow.keras.Model",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.evaluate(*args, **kwargs)

    def predict(
        self,
        generator,
        predict_data=None,
        summarise=False,
        output_layer=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        """
        This method generates predictions for the data produced by the given generator or alternatively the data
        given in parameter predict_data.

        For detailed descriptions of Keras-specific parameters consult the Keras documentation
        at https://keras.io/models/sequential/

        Args:
            generator: The generator object that, if predict_data is None, should be one of type
                GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator, GraphSAGELinkGenerator,
                or HinSAGELinkGenerator. However, if predict_data is not None, then generator should be one of type
                NodeSequence, LinkSequence, SparseFullBatchSequence, or FullBatchSequence.
            predict_data (None or iterable): If not None, then it is an iterable, e.g. list, that specifies the node IDs
                to make predictions for. If generator is of type FullBatchNodeGenerator then predict_data should be all
                the nodes in the graph since full batch approaches such as GCN and GAT can only be used to make
                predictions for all graph nodes.
            summarise (bool): If True, then the mean of the predictions over self.n_estimators and
                self.n_predictions are returned for each query point. If False, then all predictions are returned.
            output_layer (None or int): If not None, then the predictions are the outputs of the layer specified.
                The default is the model's output layer.
            max_queue_size (int): (Keras-specific parameter) The maximum size for the generator queue.
            workers (int): (Keras-specific parameter) The maximum number of workers to use.
            use_multiprocessing (bool): (Keras-specific parameter) If True then use process based threading.
            verbose (int): (Keras-specific parameter) The verbocity mode that should be 0 or 1 with the former turning
                verbocity off and the latter on.


        Returns:
            numpy array: The predictions. It will have shape `MxKxNxF` if **summarise** is set to `False`, or NxF
            otherwise. `M` is the number of estimators in the ensemble; `K` is the number of predictions per query
            point; `N` is the number of query points; and `F` is the output dimensionality of the specified layer
            determined by the shape of the output layer.

        """
        data_generator = generator
        if predict_data is not None:
            if not isinstance(
                generator,
                (
                    sg.mapper.GraphSAGENodeGenerator,
                    sg.mapper.HinSAGENodeGenerator,
                    sg.mapper.FullBatchNodeGenerator,
                ),
            ):
                raise ValueError(
                    "({}) generator parameter must be of type GraphSAGENodeGenerator, HinSAGENodeGenerator, or FullBatchNodeGenerator. Received type {}".format(
                        type(self).__name__, type(generator).__name__
                    )
                )
            data_generator = generator.flow(predict_data)
        elif not isinstance(
            generator,
            (
                sg.mapper.NodeSequence,
                sg.mapper.LinkSequence,
                sg.mapper.FullBatchSequence,
                sg.mapper.SparseFullBatchSequence,
            ),
        ):
            raise ValueError(
                "({}) If x is None, generator must be one of type NodeSequence, "
                "LinkSequence, SparseFullBatchSequence, or FullBatchSequence.".format(
                    type(self).__name__
                )
            )

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
                    model.predict(
                        data_generator,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        verbose=verbose,
                    )
                )
            # add to predictions list
            predictions.append(model_predictions)

        predictions = np.array(predictions)

        if summarise is True:
            # average the predictions across models and predictions per query point
            predictions = np.mean(predictions, axis=(0, 1))

        # if len(predictions.shape) > 4:
        #     predictions = predictions.reshape(predictions.shape[0:3] + (-1,))

        return predictions

    def predict_generator(self, *args, **kwargs):
        """
        Deprecated: use :meth:`predict`.
        """
        warnings.warn(
            "'predict_generator' has been replaced by 'predict', to match tensorflow.keras.Model",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predict(*args, **kwargs)


#
#
#
class BaggingEnsemble(Ensemble):
    """
    The BaggingEnsemble class can be used to create ensembles of stellargraph's graph neural network algorithms
    including GCN, GraphSAGE, GAT, and HinSAGE. Ensembles can be used for training classification and regression
    problems for node attribute inference and link prediction.

    This class can be used to create Bagging ensembles.

    Bagging ensembles add model diversity in two ways: (1) by random initialisation of the models' weights (before
    training) to different values; and (2) by bootstrap sampling of the training data for each model. That is, each
    model in the ensemble is trained on a random subset of the training examples, sampled with replacement from the
    original training data.
    """

    def __init__(self, model, n_estimators=3, n_predictions=3):
        """

        Args:
            model: A keras model.
            n_estimators (int):  The number of estimators (aka models) in the ensemble.
            n_predictions (int):  The number of predictions per query point per estimator
        """
        super().__init__(
            model=model, n_estimators=n_estimators, n_predictions=n_predictions
        )

    def fit(
        self,
        generator,
        train_data,
        train_targets,
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
        bag_size=None,
        use_early_stopping=False,
        early_stopping_monitor="val_loss",
    ):
        """
        This method trains the ensemble on the data given in train_data and train_targets. If validation data are
        also given, then the training metrics are evaluated on these data and results printed on screen if verbose
        level is greater than 0.

        The method trains each model in the ensemble in series for the number of epochs specified. Training can
        also stop early with the best model as evaluated on the validation data, if use_early_stopping is enabled.

        Each model in the ensemble is trained using a bootstrapped sample of the data (the train data are re-sampled
        with replacement.) The number of bootstrap samples can be specified via the bag_size parameter; by default,
        the number of bootstrap samples equals the number of training points.

        For detail descriptions of Keras-specific parameters consult the Keras documentation
        at https://keras.io/models/sequential/

        Args:
            generator: The generator object for training data. It should be one of type
                GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator, GraphSAGELinkGenerator,
                or HinSAGELinkGenerator.
            train_data (iterable): It is an iterable, e.g. list, that specifies the data
                to train the model with.
            train_targets (iterable): It is an iterable, e.g. list, that specifies the target
                values for the train data.
            steps_per_epoch (None or int): (Keras-specific parameter) If not None, it specifies the number of steps
                to yield from the generator before declaring one epoch finished and starting a new epoch.
            epochs (int): (Keras-specific parameter) The number of training epochs.
            verbose (int): (Keras-specific parameter) The verbocity mode that should be 0 , 1, or 2 meaning silent,
                progress bar, and one line per epoch respectively.
            validation_data: A generator for validation data that is optional (None). If not None then, it should
                be one of type GraphSAGENodeGenerator, HinSAGENodeGenerator, FullBatchNodeGenerator,
                GraphSAGELinkGenerator, or HinSAGELinkGenerator.
            validation_steps (None or int): (Keras-specific parameter) If validation_generator is not None, then it
                specifies the number of steps to yield from the generator before stopping at the end of every epoch.
            class_weight (None or dict): (Keras-specific parameter) If not None, it should be a dictionary
                mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during
                training only). This can be useful to tell the model to "pay more attention" to samples from an
                under-represented class.
            max_queue_size (int): (Keras-specific parameter) The maximum size for the generator queue.
            workers (int): (Keras-specific parameter) The maximum number of workers to use.
            use_multiprocessing (bool): (Keras-specific parameter) If True then use process based threading.
            shuffle (bool): (Keras-specific parameter) If True, then it shuffles the order of batches at the
                beginning of each training epoch.
            initial_epoch (int): (Keras-specific parameter) Epoch at which to start training (useful for resuming a
                previous training run).
            bag_size (None or int): The number of samples in a bootstrap sample. If None and bagging is used, then
                the number of samples is equal to the number of training points.
            use_early_stopping (bool): If set to True, then early stopping is used when training each model
                in the ensemble. The default is False.
            early_stopping_monitor (str): The quantity to monitor for early stopping, e.g., 'val_loss',
                'val_weighted_acc'. It should be a valid Keras metric.

        Returns:
            list: It returns a list of Keras History objects each corresponding to one trained model in the ensemble.

        """
        if not isinstance(
            generator,
            (
                sg.mapper.GraphSAGENodeGenerator,
                sg.mapper.HinSAGENodeGenerator,
                sg.mapper.FullBatchNodeGenerator,
                sg.mapper.GraphSAGELinkGenerator,
                sg.mapper.HinSAGELinkGenerator,
            ),
        ):
            raise ValueError(
                "({}) generator parameter must be of type GraphSAGENodeGenerator, HinSAGENodeGenerator, "
                "FullBatchNodeGenerator, GraphSAGELinkGenerator, or HinSAGELinkGenerator if you want to use Bagging. "
                "Received type {}".format(type(self).__name__, type(generator).__name__)
            )
        if bag_size is not None and (bag_size > len(train_data) or bag_size <= 0):
            raise ValueError(
                "({}) bag_size must be positive and less than or equal to the number of training points ({})".format(
                    type(self).__name__, len(train_data)
                )
            )
        if train_targets is None:
            raise ValueError(
                "({}) If train_data is given then train_targets must be given as well.".format(
                    type(self).__name__
                )
            )

        self.history = []

        num_points_per_bag = bag_size if bag_size is not None else len(train_data)

        # Prepare the training data for each model. Use sampling with replacement to create len(self.models)
        # datasets.
        for model in self.models:
            di_index = np.random.choice(
                len(train_data), size=num_points_per_bag
            )  # sample with replacement
            di_train = train_data[di_index]

            di_targets = train_targets[di_index]

            di_gen = generator.flow(di_train, di_targets)

            es_callback = None
            if use_early_stopping and validation_data is not None:
                es_callback = [
                    EarlyStopping(
                        monitor=early_stopping_monitor,
                        patience=self.early_stoppping_patience,
                        restore_best_weights=True,
                    )
                ]

            self.history.append(
                model.fit(
                    di_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=es_callback,
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

    def fit_generator(self, *args, **kwargs):
        """
        Deprecated: use :meth:`fit`.
        """
        warnings.warn(
            "'fit_generator' has been replaced by 'fit', to match tensorflow.keras.Model",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fit(*args, **kwargs)
