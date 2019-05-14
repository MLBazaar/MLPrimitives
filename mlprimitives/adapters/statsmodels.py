from statsmodels.tsa import arima_model


class ARIMA(object):
    """A Wrapper for the statsmodels ARIMA model for time series predictions"""

    def __init__(self, p, d, q, steps):
        self.p = p
        self.d = d
        self.q = q
        self.steps = steps

    def predict(self, X):
        model = arima_model.ARIMA(X, order=(self.p, self.d, self.q))
        model = model.fit(disp=0)
        y = model.forecast(self.steps)
        return y[0]
