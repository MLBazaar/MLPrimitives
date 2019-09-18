from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from mlprimitives.custom.feature_extraction import CategoricalEncoder, FeatureExtractor


class FeatureExtractorTest(TestCase):

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
            _detect_features = Mock()
            _fit = Mock()

        fe = FE(features=['b'])
        X = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': [1, 2, 3]
        })

        fe.fit(X)

        assert fe._features == ['b']
        assert fe._detect_features.not_called()

    def test_fit_auto_pandas(self):
        class FE(FeatureExtractor):
            _detect_features = Mock(return_value=['a', 'b'])
            _fit = Mock()

        fe = FE(features='auto')
        X = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': [1, 2, 3]
        })

        fe.fit(X)

        assert fe._features == ['a', 'b']
        assert fe._detect_features.called_once_with(X)
        expected_calls = [
            ((pd.Series(['a', 'b', 'c']), ), {}),
            ((pd.Series(['d', 'e', 'f']), ), {})
        ]
        self.assert_equal(expected_calls, fe._fit.call_args_list)

    def test_fit_auto_numpy(self):
        class FE(FeatureExtractor):
            _detect_features = Mock(return_value=[0, 1])
            _fit = Mock()

        fe = FE(features='auto')
        X = [
            ['a', 'd', 1],
            ['b', 'e', 2],
            ['c', 'f', 3],
        ]

        fe.fit(X)

        assert fe._features == [0, 1]
        assert fe._detect_features.called_once_with(X)
        expected_calls = [
            ((pd.Series(['a', 'b', 'c']), ), {}),
            ((pd.Series(['d', 'e', 'f']), ), {})
        ]
        self.assert_equal(expected_calls, fe._fit.call_args_list)


class CategoricalEncoderTest(TestCase):

    def test___init__(self):
        ce = CategoricalEncoder(max_labels=5, max_unique_ratio=0.5, features='auto')

        assert ce.max_labels == 5
        assert ce.max_unique_ratio == 0.5
        assert ce.features == 'auto'

    @patch('mlprimitives.custom.feature_extraction.FeatureExtractor.fit')
    def test_fit(self, fit_mock):
        """Check how self.encoders is reset, and super.fit called."""
        ce = CategoricalEncoder()
        ce.encoders = {
            'past': 'encoders'
        }

        ce.fit('some_X')

        assert ce.encoders == dict()
        fit_mock.assert_called_once_with('some_X')

    @patch('mlprimitives.custom.feature_extraction.OneHotLabelEncoder')
    def test__fit(self, ohle_mock):
        ce = CategoricalEncoder()
        ce.encoders = dict()

        x = pd.Series(['a', 'b', 'a'], name='test')
        ce._fit(x)

        assert ce.encoders['test'] == ohle_mock.return_value
        ohle_mock.return_value.fit.assert_called_once_with(x)

    def test__transform(self):
        ce = CategoricalEncoder()
        ohle_instance = Mock()
        ohle_instance.transform.return_value = pd.DataFrame({
            'test=a': [1, 0, 1],
            'test=b': [0, 1, 0],
        })
        ce.encoders = {
            'test': ohle_instance
        }

        x = pd.Series(['a', 'b', 'a'], name='test')
        returned = ce._transform(x)

        expected = pd.DataFrame({
            'test=a': [1, 0, 1],
            'test=b': [0, 1, 0],
        })
        assert expected.equals(returned)
        ohle_instance.transform.assert_called_once_with(x)

    def test__detect_features_no_max_unique(self):
        ce = CategoricalEncoder(max_unique_ratio=0)

        X = pd.DataFrame({
            'unique': ['a', 'b', 'c', 'd'],
            'not_unique': ['a', 'b', 'a', 'a'],
            'not_feature': [1, 2, 3, 4],
        })

        features = ce._detect_features(X)

        assert set(features) == {'unique', 'not_unique'}

    def test__detect_features_max_unique(self):
        ce = CategoricalEncoder(max_unique_ratio=0.5)

        X = pd.DataFrame({
            'completely_unique': ['a', 'b', 'c', 'd', 'e'],
            'too_unique': ['a', 'b', 'c', 'd', 'a'],
            'not_unique': ['a', 'b', 'a', 'a', 'a'],
            'not_feature': [1, 2, 3, 4, 5],
        })

        features = ce._detect_features(X)

        assert features == ['not_unique']

    def test__detect_features_nones(self):
        ce = CategoricalEncoder(max_unique_ratio=0.5)

        X = pd.DataFrame({
            'completely_unique': ['a', 'b', 'c', 'd', 'e', None],
            'too_unique': ['a', 'b', 'c', 'd', None, 'a'],
            'not_unique': ['a', 'b', 'a', None, 'a', 'a'],
            'not_feature': [1, 2, None, 4, 5, 6],
        })

        features = ce._detect_features(X)

        assert features == ['not_unique']

    def test__detect_features_category(self):
        ce = CategoricalEncoder(max_unique_ratio=0)

        X = pd.DataFrame({
            'unique': ['a', 'b', 'c', 'd'],
            'not_unique': ['a', 'b', 'a', 'a'],
            'category': ['a', 'b', 'a', 'b'],
            'not_feature': [1, 2, 3, 4],
        })
        X['category'] = X['category'].astype('category')

        features = ce._detect_features(X)

        assert set(features) == {'unique', 'not_unique', 'category'}
