import numpy as np
from statsmodels.tsa import arima_model


class ARIMA(object):
    """A Wrapper for the statsmodels.tsa.arima_model.ARIMA class."""

    def __init__(self, p, d, q, steps):
        """Initialize the ARIMA object.

        Args:
            p (int):
                Integer denoting the order of the autoregressive model.
            d (int):
                Integer denoting the degree of differencing.
            q (int):
                Integer denoting the order of the moving-average model.
            steps (int):
                Integer denoting the number of time steps to predict ahead.
        """
        self.p = p
        self.d = d
        self.q = q
        self.steps = steps

    def predict(self, X):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the predictions for each input sequence.
        """
        arima_results = list()
        dimensions = len(X.shape)

        if dimensions > 2:
            raise ValueError("Only 1D o 2D arrays are supported")

        if dimensions == 1 or X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)

        num_sequences = len(X)
        for sequence in range(num_sequences):
            arima = arima_model.ARIMA(X[sequence], order=(self.p, self.d, self.q))
            arima_fit = arima.fit(disp=0)
            arima_results.append(arima_fit.forecast(self.steps)[0])

        arima_results = np.asarray(arima_results)

        if dimensions == 1:
            arima_results = arima_results[0]

        return arima_results
