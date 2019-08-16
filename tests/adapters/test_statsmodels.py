from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from mlprimitives.adapters.statsmodels import ARIMA


@patch('statsmodels.tsa.arima_model.ARIMA')
def test_arima_1d(arima_mock):
    arima = ARIMA(1, 0, 0, 3)
    X = np.array([1, 2, 3, 4, 5])
    arima.predict(X)
    assert_allclose(arima_mock.call_args[0][0], [1, 2, 3, 4, 5])
    assert arima_mock.call_args[1] == {'order': (1, 0, 0)}


@patch('statsmodels.tsa.arima_model.ARMAResults.forecast')
def test_predict_1d(arima_mock):
    arima_mock.return_value = [[1, 2, 3]]

    arima = ARIMA(1, 0, 0, 3)

    X = np.array([1, 2, 3, 4, 5])
    result = arima.predict(X)

    expected = np.array([1, 2, 3])
    assert_allclose(result, expected)
    arima_mock.assert_called_once_with(3)


@patch('statsmodels.tsa.arima_model.ARIMA')
def test_arima_2d(arima_mock):
    arima = ARIMA(1, 0, 0, 3)
    X = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ])
    arima.predict(X)
    assert_allclose(arima_mock.call_args_list[0][0], [[1, 2, 3, 4, 5]])
    assert_allclose(arima_mock.call_args_list[1][0], [[6, 7, 8, 9, 10]])
    assert_allclose(arima_mock.call_args_list[2][0], [[11, 12, 13, 14, 15]])
    assert arima_mock.call_args_list[0][1] == {'order': (1, 0, 0)}
    assert arima_mock.call_args_list[1][1] == {'order': (1, 0, 0)}
    assert arima_mock.call_args_list[2][1] == {'order': (1, 0, 0)}


@patch('statsmodels.tsa.arima_model.ARMAResults.forecast')
def test_predict_2d(arima_mock):
    arima_mock.side_effect = [
        [[1, 2, 3]],
        [[4, 5, 6]],
        [[7, 8, 9]],
    ]
    arima = ARIMA(1, 0, 0, 3)

    X = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ])
    result = arima.predict(X)

    expected = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    assert_allclose(result, expected)
    arima_mock.assert_called_with(3)
