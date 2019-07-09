# -*- coding: utf-8 -*-

import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

LOGGER = logging.getLogger(__name__)


class OneHotLabelEncoder(object):
    """Combination of LabelEncoder + OneHotEncoder.

    >>> df = pd.DataFrame([
    ... {'a': 'a', 'b': 1, 'c': 1},
    ... {'a': 'a', 'b': 2, 'c': 2},
    ... {'a': 'b', 'b': 2, 'c': 1},
    ... ])
    >>> OneHotLabelEncoder().fit_transform(df.a)
       a=a  a=b
    0    1    0
    1    1    0
    2    0    1
    >>> OneHotLabelEncoder(max_labels=1).fit_transform(df.a)
       a=a
    0    1
    1    1
    2    0
    >>> OneHotLabelEncoder(name='a_name').fit_transform(df.a)
       a_name=a  a_name=b
    0         1         0
    1         1         0
    2         0         1
    """

    def __init__(self, name=None, max_labels=None):
        self.name = name
        self.max_labels = max_labels

    def fit(self, feature):
        self.dummies = pd.Series(feature.value_counts().index).astype(str)
        if self.max_labels:
            self.dummies = self.dummies[:self.max_labels]

    def transform(self, feature):
        name = self.name or feature.name
        dummies = pd.get_dummies(feature.astype(str))
        dummies = dummies.reindex(columns=self.dummies, fill_value=0)
        dummies.columns = ['{}={}'.format(name, c) for c in self.dummies]
        return dummies

    def fit_transform(self, feature):
        self.fit(feature)
        return self.transform(feature)


class FeatureExtractor(object):
    """Single FeatureExtractor applied to multiple features."""

    def __init__(self, copy=True, features=None, keep=False):
        self.copy = copy
        self.features = features or []
        self.keep = keep
        self._features = []

    def _fit(self, x):
        pass

    def _detect_feautres(self, X):
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.features == 'auto':
            self._features = self._detect_features(X)
        else:
            self._features = self.features

        for feature in self._features:
            self._fit(X[feature])

    def _transform(self, x):
        pass

    def transform(self, X):
        if self._features:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            elif self.copy:
                X = X.copy()

        for feature in self._features:
            LOGGER.debug("Extracting feature %s", feature)
            if self.keep:
                x = X[feature]
            else:
                x = X.pop(feature)

            extracted = self._transform(x)
            X = pd.concat([X, extracted], axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def _is_str(x):
    return isinstance(x, str)


class CategoricalEncoder(FeatureExtractor):
    """Use the OneHotLabelEncoder only on categorical features.

    >>> df = pd.DataFrame([
    ... {'a': 'a', 'b': 1, 'c': 1},
    ... {'a': 'a', 'b': 2, 'c': 2},
    ... {'a': 'b', 'b': 2, 'c': 1},
    ... ])
    >>> ce = CategoricalEncoder()
    >>> ce.fit_transform(df, categorical_features=['a', 'c'])
       b  a=a  a=b  c=1  c=2
    0  1    1    0    1    0
    1  2    1    0    0    1
    2  2    0    1    1    0
    """

    def __init__(self, max_labels=None, max_unique_ratio=1, **kwargs):
        self.max_labels = max_labels
        self.max_unique_ratio = max_unique_ratio
        super(CategoricalEncoder, self).__init__(**kwargs)

    def _detect_features(self, X):
        features = list()

        for column in X.select_dtypes('object'):
            x = X[column]
            unique_ratio = len(x.unique()) / len(x)
            if unique_ratio < self.max_unique_ratio:
                if x.apply(_is_str).all():
                    features.append(column)

        return features

    def fit(self, X, y=None):
        self.encoders = dict()
        super(CategoricalEncoder, self).fit(X)

    def _fit(self, x):
        encoder = OneHotLabelEncoder(x.name, self.max_labels)
        encoder.fit(x)
        self.encoders[x.name] = encoder

    def _transform(self, x):
        encoder = self.encoders[x.name]
        return encoder.transform(x)


class StringVectorizer(FeatureExtractor):
    """Use the sklearn CountVectorizer only on string features."""

    DTYPE = 'object'

    def __init__(self, copy=True, features=None, keep=False, min_words=3, **kwargs):
        self.kwargs = kwargs
        self.min_words = min_words
        super(StringVectorizer, self).__init__(copy, features, keep)

    def _detect_features(self, X):
        features = []

        analyzer = CountVectorizer(**self.kwargs).build_analyzer()
        for column in X.select_dtypes('object'):
            try:
                if (X[column].apply(analyzer).str.len() >= self.min_words).any():
                    features.append(column)
            except (ValueError, AttributeError):
                pass

        return features

    def fit(self, X, y=None):
        self.vectorizers = dict()
        super(StringVectorizer, self).fit(X)

    def _fit(self, x):
        vectorizer = CountVectorizer(**self.kwargs)
        vectorizer.fit(x.fillna('').astype(str))
        self.vectorizers[x.name] = vectorizer

    def _transform(self, x):
        vectorizer = self.vectorizers[x.name]
        bow = vectorizer.transform(x.fillna('').astype(str))
        bow_columns = ['{}_{}'.format(x.name, f) for f in vectorizer.get_feature_names()]
        return pd.DataFrame(bow.toarray(), columns=bow_columns, index=x.index)


class DatetimeFeaturizer(FeatureExtractor):
    """Extract features from a datetime."""

    def _detect_features(self, X):
        return list(X.select_dtypes('datetime').columns)

    def _transform(self, x):
        prefix = x.name + '_'
        features = {
            prefix + 'year': x.dt.year,
            prefix + 'month': x.dt.month,
            prefix + 'day': x.dt.day,
            prefix + 'weekday': x.dt.day,
            prefix + 'hour': x.dt.hour,
        }
        return pd.DataFrame(features)
