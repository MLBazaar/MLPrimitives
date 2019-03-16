
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from mlprimitives.custom.timeseries_anomalies import find_anomalies, find_sequences


class FindSequencesTest(TestCase):

    THRESHOLD = 0.5

    def _run(self, errors, expected):
        found = find_sequences(np.asarray(errors), self.THRESHOLD)

        assert found == expected

    def test_find_sequences_no_sequences(self):
        self._run([0, 0, 0, 0], [])

    def test_find_sequences_all_one_sequence(self):
        self._run([1, 1, 1, 1], [(0, 3)])

    def test_find_sequences_open_end(self):
        self._run([0, 1, 1, 1], [(1, 3)])

    def test_find_sequences_open_start(self):
        self._run([1, 1, 1, 0], [(0, 2)])

    def test_find_sequences_middle(self):
        self._run([0, 1, 1, 0], [(1, 2)])

    def test_find_sequences_stop_length_one(self):
        self._run([1, 0, 1, 1], [(0, 0), (2, 3)])

    def test_find_sequences_open_length_one(self):
        self._run([1, 0, 0, 1], [(0, 0), (3, 3)])


class FindAnomaliesTest(TestCase):

    THRESHOLD = 0.5
    INDEX = [10, 11, 12, 13]

    def _run(self, errors, expected):
        found = find_anomalies(np.asarray(errors), self.INDEX)

        assert_allclose(found, expected)

    def test_find_anomalies_no_anomalies(self):
        self._run([0, 0, 0, 0], [])

    def test_find_anomalies_one_anomaly(self):
        self._run([0, 0.5, 0.5, 0], [[11., 12., 0.025]])

    def test_find_anomalies_open_start(self):
        self._run([0.5, 0.5, 0, 0], [[10., 11., 0.025]])

    def test_find_anomalies_open_end(self):
        self._run([0, 0, 0.5, 0.5], [[12., 13., 0.025]])

    def test_find_anomalies_two_anomalies(self):
        self._run([0.5, 0, 0.5, 0], [[10., 10., 0.025], [12., 12., 0.025]])
        self._run([0., 0.5, 0., 0.5], [[11., 11., 0.025], [13., 13., 0.025]])
