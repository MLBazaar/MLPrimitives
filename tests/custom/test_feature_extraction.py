from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from mlprimitives.custom.feature_extraction import FeatureExtractor


class FeatureExtractorTest(TestCase):

    def test_detect_features(self):
        X = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': [1, 2, 3]
        })

        features = FeatureExtractor.detect_features(X)

        assert features == ['a', 'b']

    @classmethod
    def assert_equal(cls, obj1, obj2):
        if hasattr(obj1, 'equals'):
            assert obj1.equals(obj2)
        elif hasattr(obj1, '__len__'):
            assert len(obj1) == len(obj2)
            for el1, el2 in zip(obj1, obj2):
                cls.assert_equal(el1, el2)

        else:
            assert obj1 == obj2

    def test_fit_features(self):
        class FE(FeatureExtractor):
            detect_features = Mock()
            _fit = Mock()

        fe = FE(features=['b'])
        X = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': [1, 2, 3]
        })

        fe.fit(X)

        assert fe._features == ['b']
        assert fe.detect_features.not_called()

    def test_fit_auto_pandas(self):
        class FE(FeatureExtractor):
            detect_features = Mock(return_value=['a', 'b'])
            _fit = Mock()

        fe = FE(features='auto')
        X = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': [1, 2, 3]
        })

        fe.fit(X)

        assert fe._features == ['a', 'b']
        assert fe.detect_features.called_once_with(X)
        expected_calls = [
            ((pd.Series(['a', 'b', 'c']), ), {}),
            ((pd.Series(['d', 'e', 'f']), ), {})
        ]
        self.assert_equal(expected_calls, fe._fit.call_args_list)

    def test_fit_auto_numpy(self):
        class FE(FeatureExtractor):
            detect_features = Mock(return_value=[0, 1])
            _fit = Mock()

        fe = FE(features='auto')
        X = [
            ['a', 'd', 1],
            ['b', 'e', 2],
            ['c', 'f', 3],
        ]

        fe.fit(X)

        assert fe._features == [0, 1]
        assert fe.detect_features.called_once_with(X)
        expected_calls = [
            ((pd.Series(['a', 'b', 'c']), ), {}),
            ((pd.Series(['d', 'e', 'f']), ), {})
        ]
        self.assert_equal(expected_calls, fe._fit.call_args_list)
