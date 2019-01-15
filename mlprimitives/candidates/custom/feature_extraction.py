# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from mlprimitives.custom.feature_extraction import FeatureExtractor


class DatetimeFeaturizer(FeatureExtractor):
    """Extract features from a datetime."""

    def detect_features(self, X):
        features = []
        for column in X.columns:
            if np.issubdtype(X[column].dtype, np.datetime64):
                features.append(column)

        return features

    def _transform(self, x):
        if not np.issubdtype(x.dtype, np.datetime64):
            x = pd.to_datetime(x)

        prefix = x.name + '_'
        features = {
            prefix + 'year': x.dt.year,
            prefix + 'month': x.dt.month,
            prefix + 'day': x.dt.day,
            prefix + 'weekday': x.dt.day,
            prefix + 'hour': x.dt.hour,
        }
        return pd.DataFrame(features)
