# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder


class ClassEncoder():

    def __init__(self):
        self._label_encoder = LabelEncoder()

    def fit(self, y):
        self._label_encoder.fit(y)

    def encode(self, y):
        if y is not None:
            classes = self._label_encoder.classes_
            y = self._label_encoder.transform(y)
            return y, classes


class ClassDecoder():

    def __init__(self):
        self._label_encoder = LabelEncoder()

    def fit(self, classes):
        self._label_encoder.classes_ = classes

    def decode(self, y):
        return self._label_encoder.inverse_transform(y)


class RangeScaler():

    _data_min = None
    _data_scale = None
    _data_range = None

    def __init__(self, out_min, out_max):
        self._out_min = out_min
        self._out_scale = out_max - out_min

    def fit(self, X):
        data_max = X.max(axis=0)
        self._data_min = X.min(axis=0)
        self._data_scale = data_max - self._data_min
        self._data_range = (self._data_min, data_max)

    def scale(self, X):
        scaled = (X - self._data_min) / self._data_scale
        rescaled = (scaled * self._out_scale) + self._out_min

        return rescaled, self._data_range


class RangeUnscaler():

    def __init__(self, out_min, out_max):
        self._out_min = out_min
        self._out_scale = out_max - out_min

    def fit(self, data_range):
        self._data_min = data_range[0]
        self._data_scale = data_range[1] - self._data_min

    def unscale(self, X):
        unscaled = (X - self._out_min) / self._out_scale
        return (unscaled * self._data_scale) + self._data_min
