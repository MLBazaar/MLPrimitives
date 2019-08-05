import numpy as np
from statsmodels.tsa import arima_model


class ARIMA(object):
    """A Wrapper for the statsmodels ARIMA model for time series predictions"""

    def __init__(self, p, d, q, steps):
        self.p = p
        self.d = d
        self.q = q
        self.steps = steps

    def predict(self, X):
        y = list()
        if len(X.shape) == 1 or len(X.shape) == 2 and X.shape[1] == 1:
            X = np.expand_dims(X, axis=0)
        for i in range(len(X)):
            model = arima_model.ARIMA(X[i], order=(self.p, self.d, self.q))
            model = model.fit(disp=0)
            y.append(model.forecast(self.steps)[0])
        return np.asarray(y)
