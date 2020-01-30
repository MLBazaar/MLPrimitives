import unittest

import numpy as np

from mlprimitives.custom.preprocessing import RangeScaler, RangeUnscaler


class RangeScalerTest(unittest.TestCase):

    def test_fit(self):
        scaler = RangeScaler(0, 1)

        data = np.array([
            [1, 2],
            [2, 4],
            [3, 6],
            [2, 8],
            [3, 10],
        ])
        scaler.fit(data)

        np.testing.assert_equal(scaler._data_min, np.array([1, 2]))
        np.testing.assert_equal(scaler._data_scale, np.array([2, 8]))
        np.testing.assert_equal(scaler._data_range, (np.array([1, 2]), np.array([3, 10])))

    def test_scale(self):
        scaler = RangeScaler(0, 1)
        scaler._data_min = np.array([1, 2])
        scaler._data_scale = np.array([2, 8])
        scaler._data_range = (np.array([1, 2]), np.array([3, 10]))

        data = np.array([
            [1, 2],
            [2, 4],
            [3, 6],
            [2, 8],
            [3, 10]
        ])
        scaled = scaler.scale(data)

        expected = (
            np.array([
                [0, 0],
                [0.5, 0.25],
                [1, 0.5],
                [0.5, 0.75],
                [1, 1],
            ]),
            (np.array([1, 2]), np.array([3, 10]))
        )
        np.testing.assert_equal(scaled, expected)


class RangeiUnscalerTest(unittest.TestCase):

    def test_fit(self):
        unscaler = RangeUnscaler(0, 1)

        data_range = (np.array([1, 2]), np.array([3, 10]))
        unscaler.fit(data_range)

        np.testing.assert_equal(unscaler._data_min, np.array([1, 2]))
        np.testing.assert_equal(unscaler._data_scale, np.array([2, 8]))

    def test_unscale(self):
        unscaler = RangeUnscaler(0, 1)
        unscaler._data_min = np.array([1, 2])
        unscaler._data_scale = np.array([2, 8])

        data = np.array([
            [0, 0],
            [0.5, 0.25],
            [1, 0.5],
            [0.5, 0.75],
            [1, 1],
        ])
        unscaled = unscaler.unscale(data)

        expected = np.array([
            [1, 2],
            [2, 4],
            [3, 6],
            [2, 8],
            [3, 10]
        ])
        np.testing.assert_equal(unscaled, expected)
