from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from mlprimitives.custom.timeseries_preprocessing import (
    intervals_to_mask, rolling_window_sequences, time_segments_aggregate, time_segments_average)


class IntervalsToMaskTest(TestCase):

    def _run(self, index, intervals, expected):
        mask = intervals_to_mask(index, intervals)

        assert_allclose(mask, expected)

    def test_no_intervals(self):
        index = np.array([1, 2, 3, 4])
        intervals = None
        expected = np.array([False, False, False, False])
        self._run(index, intervals, expected)

    def test_empty_list(self):
        index = np.array([1, 2, 3, 4])
        intervals = list()
        expected = np.array([False, False, False, False])
        self._run(index, intervals, expected)

    def test_empty_array(self):
        index = np.array([1, 2, 3, 4])
        intervals = np.array([])
        expected = np.array([False, False, False, False])
        self._run(index, intervals, expected)

    def test_one_interval(self):
        index = np.array([1, 2, 3, 4])
        intervals = np.array([[2, 3]])
        expected = np.array([False, True, True, False])
        self._run(index, intervals, expected)

    def test_two_intervals(self):
        index = np.array([1, 2, 3, 4, 5, 6, 7])
        intervals = np.array([[2, 3], [5, 6]])
        expected = np.array([False, True, True, False, True, True, False])
        self._run(index, intervals, expected)

    def test_two_intervals_list(self):
        index = np.array([1, 2, 3, 4, 5, 6, 7])
        intervals = [[2, 3], [5, 6]]
        expected = np.array([False, True, True, False, True, True, False])
        self._run(index, intervals, expected)

    def test_start_index(self):
        index = np.array([1, 2, 3, 4])
        intervals = [[1, 2]]
        expected = np.array([True, True, False, False])
        self._run(index, intervals, expected)

    def test_end_index(self):
        index = np.array([1, 2, 3, 4])
        intervals = [[3, 4]]
        expected = np.array([False, False, True, True])
        self._run(index, intervals, expected)

    def test_whole_index(self):
        index = np.array([1, 2, 3, 4])
        intervals = [[1, 4]]
        expected = np.array([True, True, True, True])
        self._run(index, intervals, expected)

    def test_exceed_index_start(self):
        index = np.array([2, 3, 4])
        intervals = [[1, 3]]
        expected = np.array([True, True, False])
        self._run(index, intervals, expected)

    def test_exceed_index_end(self):
        index = np.array([2, 3, 4])
        intervals = [[3, 5]]
        expected = np.array([False, True, True])
        self._run(index, intervals, expected)

    def test_exceed_index(self):
        index = np.array([2, 3, 4])
        intervals = [[1, 5]]
        expected = np.array([True, True, True])
        self._run(index, intervals, expected)


class RollingWindowSequencesTest(TestCase):

    def _run(self, X, index, expected_X, expected_y, expected_X_index, expected_y_index,
             window_size=2, target_size=1, step_size=1, target_column=0, drop=None,
             drop_windows=False):
        X, y, X_index, y_index = rolling_window_sequences(X, index, window_size, target_size,
                                                          step_size, target_column, drop,
                                                          drop_windows)
        assert_allclose(X.astype(float), expected_X)
        assert_allclose(y.astype(float), expected_y)
        assert_allclose(X_index, expected_X_index)
        assert_allclose(y_index, expected_y_index)

    def test_no_drop(self):
        X = np.array([[0.5], [1], [0.5], [1]])
        index = np.array([1, 2, 3, 4])
        expected_X = np.array([[[0.5], [1]], [[1], [0.5]]])
        expected_y = np.array([[0.5], [1]])
        expected_X_index = np.array([1, 2])
        expected_y_index = np.array([3, 4])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index)

    def test_drop_mask(self):
        X = np.array([[0.5], [1], [0.5], [1], [0.5], [1], [0.5], [1], [0.5]])
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        drop = np.array([False, False, False, True, True, False, False, False, False])
        expected_X = np.array([[[0.5], [1]], [[1], [0.5]], [[0.5], [1]]])
        expected_y = np.array([[0.5], [1], [0.5]])
        expected_X_index = np.array([1, 6, 7])
        expected_y_index = np.array([3, 8, 9])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)

    def test_drop_float(self):
        X = np.array([[0.5], [0.5], [0.5], [1.0], [1.0], [0.5], [0.5], [0.5]])
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        drop = 1.0
        expected_X = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        expected_y = np.array([[0.5], [0.5]])
        expected_X_index = np.array([1, 6])
        expected_y_index = np.array([3, 8])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)

    def test_drop_None(self):
        X = np.array([[0.5], [0.5], [0.5], [None], [None], [0.5], [0.5], [0.5]])
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        drop = None
        expected_X = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        expected_y = np.array([[0.5], [0.5]])
        expected_X_index = np.array([1, 6])
        expected_y_index = np.array([3, 8])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)

    def test_drop_float_nan(self):
        X = np.array([[0.5], [0.5], [0.5], ['nan'], ['nan'], [0.5], [0.5], [0.5]]).astype(float)
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        drop = float('nan')
        expected_X = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        expected_y = np.array([[0.5], [0.5]])
        expected_X_index = np.array([1, 6])
        expected_y_index = np.array([3, 8])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)

    def test_drop_str(self):
        X = np.array([[0.5], [0.5], [0.5], ['test'], ['test'], [0.5], [0.5], [0.5]])
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        drop = "test"
        expected_X = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        expected_y = np.array([[0.5], [0.5]])
        expected_X_index = np.array([1, 6])
        expected_y_index = np.array([3, 8])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)

    def test_drop_bool(self):
        X = np.array([[0.5], [0.5], [0.5], [False], [False], [0.5], [0.5], [0.5]])
        index = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        drop = False
        expected_X = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        expected_y = np.array([[0.5], [0.5]])
        expected_X_index = np.array([1, 6])
        expected_y_index = np.array([3, 8])
        self._run(X, index, expected_X, expected_y, expected_X_index, expected_y_index,
                  drop=drop, drop_windows=True)


class TimeSegmentsAverageTest(TestCase):

    def _run(self, X, interval, expected_values, expected_index, time_column):
        values, index = time_segments_average(X, interval, time_column)

        assert_allclose(values, expected_values)
        assert_allclose(index, expected_index)

    def test_array(self):
        X = np.array([[1, 1], [2, 3], [3, 1], [4, 3]])
        interval = 2
        expected_values = np.array([[2], [2]])
        expected_index = np.array([1, 3])
        self._run(X, interval, expected_values, expected_index, time_column=0)

    def test_pandas_dataframe(self):
        X = pd.DataFrame([
            [1, 1],
            [2, 3],
            [3, 1],
            [4, 3]
        ], columns=['timestamp', 'value'])
        interval = 2
        expected_values = np.array([[2], [2]])
        expected_index = np.array([1, 3])
        self._run(X, interval, expected_values, expected_index, time_column="timestamp")


class TimeSegmentsAggregateTest(TestCase):

    def _run(self, X, interval, expected_values, expected_index, time_column, method=['mean']):
        values, index = time_segments_aggregate(X, interval, time_column, method=method)

        assert_allclose(values, expected_values)
        assert_allclose(index, expected_index)

    def test_array(self):
        X = np.array([[1, 1], [2, 3], [3, 1], [4, 3]])
        interval = 2
        expected_values = np.array([[2], [2]])
        expected_index = np.array([1, 3])
        self._run(X, interval, expected_values, expected_index, time_column=0)

    def test_pandas_dataframe(self):
        X = pd.DataFrame([
            [1, 1],
            [2, 3],
            [3, 1],
            [4, 3]
        ], columns=['timestamp', 'value'])
        interval = 2
        expected_values = np.array([[2], [2]])
        expected_index = np.array([1, 3])
        self._run(X, interval, expected_values, expected_index, time_column="timestamp")

    def test_multiple(self):
        X = np.array([[1, 1], [2, 3], [3, 1], [4, 3]])
        interval = 2
        expected_values = np.array([[2, 2], [2, 2]])
        expected_index = np.array([1, 3])
        self._run(X, interval, expected_values, expected_index, time_column=0,
                  method=['mean', 'median'])
