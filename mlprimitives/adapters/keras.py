# -*- coding: utf-8 -*-

import logging
import tempfile

import keras
import numpy as np

from mlprimitives.utils import import_object

LOGGER = logging.getLogger(__name__)


def build_layer(layer, hyperparameters):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    if issubclass(layer_class, keras.layers.wrappers.Wrapper):
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)


class Sequential(object):
    """A Wrapper around Sequential Keras models with a simpler interface."""

    def __getstate__(self):
        state = self.__dict__.copy()

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(state.pop('model'), fd.name, overwrite=True)
            state['model_str'] = fd.read()

        return state

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state.pop('model_str'))
            fd.flush()

            state['model'] = keras.models.load_model(fd.name)

        self.__dict__ = state

    def _build_model(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        model = keras.models.Sequential()

        for layer in self.layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        model.compile(loss=self.loss, optimizer=self.optimizer(), metrics=self.metrics)
        return model

    def __init__(self, layers, loss, optimizer, classification, callbacks=tuple(),
                 metrics=None, epochs=10, verbose=False, validation_split=0, batch_size=32,
                 shuffle=True, **hyperparameters):

        self.layers = layers
        self.optimizer = import_object(optimizer)
        self.loss = import_object(loss)
        self.metrics = metrics

        self.epochs = epochs
        self.verbose = verbose
        self.classification = classification
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle

        for callback in callbacks:
            callback['class'] = import_object(callback['class'])

        self.callbacks = callbacks

    def _setdefault(self, kwargs, key, value):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    def _augment_hyperparameters(self, X, kwargs):
        shape = np.asarray(X)[0].shape
        length = shape[0]
        self._setdefault(kwargs, 'input_shape', shape)
        self._setdefault(kwargs, 'input_dim', length)
        self._setdefault(kwargs, 'input_length', length)

        return kwargs

    def fit(self, X, y, **kwargs):
        self._augment_hyperparameters(X, kwargs)
        self.model = self._build_model(**kwargs)

        if self.classification:
            y = keras.utils.to_categorical(y)

        callbacks = [
            callback['class'](**callback.get('args', dict()))
            for callback in self.callbacks
        ]

        self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose, callbacks=callbacks,
                       validation_split=self.validation_split, batch_size=self.batch_size,
                       shuffle=self.shuffle)

    def predict(self, X):
        y = self.model.predict(X, batch_size=self.batch_size, verbose=self.verbose)

        if self.classification:
            y = np.argmax(y, axis=1)

        return y
