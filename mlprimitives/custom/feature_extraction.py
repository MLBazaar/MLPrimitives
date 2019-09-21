# -*- coding: utf-8 -*-

import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

LOGGER = logging.getLogger(__name__)


class OneHotLabelEncoder(object):
    """Combination of LabelEncoder + OneHotEncoder.

    Args:
        name (str or None):
            Name of this feature. If ``None`` is given, the name is taken
            from the training feature column.
        max_labels (int or None):
            Maximum number of columns to generate by feature.
        dropna (bool):
            Whether to drop null values before fitting. Defaults to True.

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

    def __init__(self, name=None, max_labels=None, dropna=True):
        self.name = name
        self.max_labels = max_labels
        self.dropna = dropna

    def fit(self, x):
        if self.dropna:
            x = x.dropna()

        self.dummies = pd.Series(x.value_counts().index).astype(str)
        if self.max_labels:
            self.dummies = self.dummies[:self.max_labels]

    def transform(self, x):
        name = self.name or x.name
        dummies = pd.get_dummies(x.astype(str))
        dummies = dummies.reindex(columns=self.dummies, fill_value=0)
        dummies.columns = ['{}={}'.format(name, c) for c in self.dummies]
        return dummies

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class FeatureExtractor(object):
    """Extract Features by applying single column feature extracts on multiple columns.

    Optionally detect the features on which to apply the feature extractor automatically.

    Args:
        copy (bool):
            Whether to make a copy of the input data or modify it in place.
            Defaults to ``True``.
        features (list or str):
            List of features to apply the feature extractor to. If ``'auto'`` is passed,
            try to detect the feature automatically. Defaults to an empty list.
        keep (bool):
            Whether to keep the original features instead of replacing them.
            Defaults to ``False``.
    """

    def __init__(self, copy=True, features=None, keep=False):
        self.copy = copy
        self.features = list() if features is None else features
        self.keep = keep
        self._features = list()

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


class CategoricalEncoder(FeatureExtractor):
    """FeatureExtractor that encodes categorical features using OneHotLabelEncoder.

    When autodetecting features, only features with dtype ``category`` or ``object``
    are considered.

    Optionally, a ``max_unique_ratio`` can be passed, which allows ignoring features
    that have a high number of unique values, such as primary keys.

    Args:
        max_labels (int or None):
            Maximum number of labels to use by feature. Defaults to ``None``.
        max_unique_ratio (int):
            Max proportion of unique values that a feature must have in order
            to be considered a categorical feature. If ``0`` is given, the ratio is ignored.
            Defaults to ``0``.
        dropna (bool):
            Whether to drop null values before analyzing the features and fitting
            the encoders.

    >>> df = pd.DataFrame([
    ... {'a': 'a', 'b': 1, 'c': 1},
    ... {'a': 'a', 'b': 2, 'c': 2},
    ... {'a': 'b', 'b': 2, 'c': 1},
    ... ])
    >>> df['c'] = d['c'].astype('category')
    >>> ce = CategoricalEncoder(features='auto')
    >>> ce.fit_transform(df)
       b  a=a  a=b  c=1  c=2
    0  1    1    0    1    0
    1  2    1    0    0    1
    2  2    0    1    1    0
    """

    def __init__(self, max_labels=None, max_unique_ratio=0, dropna=True, **kwargs):
        self.max_labels = max_labels
        self.max_unique_ratio = max_unique_ratio
        self.dropna = dropna
        super(CategoricalEncoder, self).__init__(**kwargs)

    def _detect_features(self, X):
        features = list()

        columns = X.select_dtypes(('object', 'category')).columns
        if not self.max_unique_ratio:
            return list(columns)

        for column in columns:
            x = X[column]
            if self.dropna:
                x = x.dropna()

            unique_ratio = len(x.unique()) / len(x)
            if unique_ratio < self.max_unique_ratio:
                features.append(column)

        return features

    def fit(self, X, y=None):
        self.encoders = dict()
        super(CategoricalEncoder, self).fit(X)

    def _fit(self, x):
        encoder = OneHotLabelEncoder(x.name, self.max_labels, self.dropna)
        encoder.fit(x)
        self.encoders[x.name] = encoder

    def _transform(self, x):
        encoder = self.encoders[x.name]
        return encoder.transform(x)


class StringVectorizer(FeatureExtractor):
    """FeatureExtractor that encodes text features using a scikit-learn CountVectorizer.

    When autodetecting features, only features with dtype ``object`` features are considered.

    Optionally, a ``min_words`` can be passed, which allows ignoring features
    have less than the given value of words in all their occurrences.

    Args:
        copy (bool):
            Whether to make a copy of the input data or modify it in place.
            Defaults to ``True``.
        features (list or str):
            List of features to apply the feature extractor to. If ``'auto'`` is passed,
            try to detect the feature automatically. Defaults to an empty list.
        keep (bool):
            Whether to keep the original features instead of replacing them.
            Defaults to ``False``.
        min_words (int):
            Minimum number of words that the features needs to have in order to be
            considered a text column.
        **kwargs:
            Any additional keywords arguments will be passed to the underlying
            StringVectorizer instances.
    """

    def __init__(self, copy=True, features=None, keep=False, min_words=0, **kwargs):
        self.kwargs = kwargs
        self.min_words = min_words
        super(StringVectorizer, self).__init__(copy, features, keep)

    def _detect_features(self, X):
        columns = X.select_dtypes('object').columns
        if not self.min_words:
            return list(columns)

        features = []
        analyzer = CountVectorizer(**self.kwargs).build_analyzer()
        for column in columns:
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
