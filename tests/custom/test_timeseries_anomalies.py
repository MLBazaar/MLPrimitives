from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from mlprimitives.custom.timeseries_anomalies import (
    _find_sequences, _get_max_errors, _prune_anomalies, find_anomalies)


class GetMaxErrorsTest(TestCase):

    MAX_BELOW = 0.1

    def _run(self, errors, sequences, expected):
        sequences = _get_max_errors(errors, sequences, self.MAX_BELOW)

        assert_frame_equal(sequences, expected)

    def test_no_anomalies(self):
        errors = np.array([0.1, 0.0, 0.1, 0.0])
        sequences = np.ndarray((0, 2))
        expected = pd.DataFrame([
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)

    def test_one_sequence(self):
        errors = np.array([0.1, 0.2, 0.2, 0.1])
        sequences = np.array([
            [1, 2]
        ])
        expected = pd.DataFrame([
            [0.2, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)

    def test_two_sequences(self):
        errors = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.2, 0.1])
        sequences = np.array([
            [1, 3],
            [5, 6]
        ])
        expected = pd.DataFrame([
            [0.3, 1, 3],
            [0.2, 5, 6],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        self._run(errors, sequences, expected)


class PruneAnomaliesTest(TestCase):

    MIN_PERCENT = 0.2

    def _run(self, max_errors, expected):
        sequences = _prune_anomalies(max_errors, self.MIN_PERCENT)

        assert_allclose(sequences, expected)

    def test_no_sequences(self):
        max_errors = pd.DataFrame([
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.ndarray((0, 2))
        self._run(max_errors, expected)

    def test_no_anomalies(self):
        max_errors = pd.DataFrame([
            [0.11, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.ndarray((0, 2))
        self._run(max_errors, expected)

    def test_one_anomaly(self):
        max_errors = pd.DataFrame([
            [0.2, 1, 2],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2]
        ])
        self._run(max_errors, expected)

    def test_two_anomalies(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.2, 4, 5],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2],
            [4, 5]
        ])
        self._run(max_errors, expected)

    def test_two_out_of_three(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.22, 4, 5],
            [0.11, 7, 8],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2],
            [4, 5]
        ])
        self._run(max_errors, expected)

    def test_two_with_a_gap(self):
        max_errors = pd.DataFrame([
            [0.3, 1, 2],
            [0.21, 4, 5],
            [0.2, 7, 8],
            [0.1, -1, -1]
        ], columns=['max_error', 'start', 'stop'])
        expected = np.array([
            [1, 2],
            [4, 5],
            [7, 8]
        ])
        self._run(max_errors, expected)


class FindSequencesTest(TestCase):

    THRESHOLD = 0.5

    def _run(self, errors, expected, expected_max):
        found, max_below = _find_sequences(np.asarray(errors), self.THRESHOLD)

        np.testing.assert_array_equal(found, expected)
        assert max_below == expected_max

    def test__find_sequences_no_sequences(self):
        self._run([0.1, 0.2, 0.3, 0.4], np.ndarray((0, 2)), 0.4)

    def test__find_sequences_all_one_sequence(self):
        self._run([1, 1, 1, 1], [(0, 3)], 0)

    def test__find_sequences_open_end(self):
        self._run([0, 1, 1, 1], [(1, 3)], 0)

    def test__find_sequences_open_start(self):
        self._run([1, 1, 1, 0], [(0, 2)], 0)

    def test__find_sequences_middle(self):
        self._run([0, 1, 1, 0], [(1, 2)], 0)

    def test__find_sequences_stop_length_one(self):
        self._run([1, 0, 1, 1], [(0, 0), (2, 3)], 0)

    def test__find_sequences_open_length_one(self):
        self._run([1, 0, 0, 1], [(0, 0), (3, 3)], 0)


class FindAnomaliesTest(TestCase):

    THRESHOLD = 0.5
    INDEX = [10, 11, 12, 13]

    def _run(self, errors, expected):
        found = find_anomalies(np.asarray(errors), self.INDEX)

        assert_allclose(found, expected)

    def test_find_anomalies_no_anomalies(self):
        self._run([0, 0, 0, 0], np.array([]))

    def test_find_anomalies_one_anomaly(self):
        self._run([0, 0.5, 0.5, 0], np.array([[11., 12., 0.025]]))

    def test_find_anomalies_open_start(self):
        self._run([0.5, 0.5, 0, 0], np.array([[10., 11., 0.025]]))

    def test_find_anomalies_open_end(self):
        self._run([0, 0, 0.5, 0.5], np.array([[12., 13., 0.025]]))

    def test_find_anomalies_two_anomalies(self):
        self._run([0.5, 0, 0.5, 0], np.array([[10., 10., 0.025], [12., 12., 0.025]]))
        self._run([0., 0.5, 0., 0.5], np.array([[11., 11., 0.025], [13., 13., 0.025]]))
