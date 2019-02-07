import numpy as np


def next_power_of_2(x):
    """Finds the next power of 2 value

    Args:
        x: Input value

    Returns:
        power_of_2: Next power of 2 value

    """

    power_of_2 = 1 if x == 0 else 2 ** np.ceil(np.log2(x))
    return power_of_2


class SpectralMask:
    """Anomalies detection in satellite telemetry data using a spectral mask"""

    def __init__(self, method='std_dev', gain=1, window_length=128, beta=4):
        """Inits SpectralMask Class with default values for method, gain, window_length, and
             beta

        Args:
            method: Method used to calculate the Spectral Mask: 'std_dev' uses the standard
              deviation of each frequency component and 'min_max' uses the minimum and maximum
              values
            gain: Multiplication factor used in the comparison between the spectral mask defined
              using the training data and the Fourier transform of the telemetry data vector
            window_length: Length of the sliding window in number of samples
            beta: Beta value for Kaiser window design

        """

        self.gain = gain
        self.beta = beta
        self.window_length = window_length
        self.window = None
        self.method = method
        self.window_length = next_power_of_2(window_length)

    def window_design(self, window_length=None, beta=None):
        """Kaiser window design

        Args:
            window_length: Length of the window in number of samples
            beta: Beta value for Kaiser window design

        Returns:
            window: Window designed using the beta and length provided as inputs

        """

        if window_length is None:
            window_length = self.window_length
        else:
            self.window_length = window_length

        if beta is None:
            beta = self.beta
        else:
            self.beta = beta

        self.window = np.kaiser(window_length, beta)

        return self.window

    def fit(self, X, method=None, gain=None, window_length=None, beta=None):
        """Defines a spectral mask based on training data

        Args:
            X: Training data
            method: Method used to calculate the Spectral Mask: 'std_dev' uses the standard
              deviation of each frequency component and 'min_max' uses the minimum and maximum
              values
            gain: Multiplication factor used in the comparison between the spectral mask defined
              using the training data and the Fourier transform of the telemetry data vector
            window_length: Length of the sliding window in number of samples
            beta: Beta value for Kaiser window design

        """

        training_signal = X

        if gain is None:
            gain = self.gain
        else:
            self.gain = gain

        if window_length is None:
            window_length = self.window_length
        else:
            self.window_length = window_length

        if beta is None:
            beta = self.beta
        else:
            self.beta = beta

        if method is None:
            method = self.method
        else:
            self.method = method

        self.window_design(window_length, beta)

        if method == 'std_dev':
            self.fit_freq_std_dev(training_signal)
        else:
            self.fit_freq_min_max(training_signal)

    def fit_freq_min_max(self, training_signal):
        """Defines a spectral mask based on training data using min and max values of each
             frequency component

        Args:
            training_signal: Training data

        """

        window_length = len(self.window)
        window_weight = sum(self.window)
        max_mask = np.zeros(int(window_length / 2) + 1)
        min_mask = np.zeros(int(window_length / 2) + 1)

        for i in range(0, len(training_signal) - window_length - 1):
            temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window)) \
                / window_weight
            max_mask = np.maximum(max_mask, temp)
            min_mask = np.minimum(min_mask, temp)

        self.mask_top = self.gain * max_mask
        self.mask_bottom = min_mask / self.gain

    def fit_freq_std_dev(self, training_signal):
        """Defines a spectral mask based on training data using the standard deviation values of
             each frequency component

        Args:
            training_signal: Training data

        """

        window_length = len(self.window)
        window_weight = sum(self.window)
        num_of_windows = len(training_signal) - window_length - 1
        mean = np.zeros(int(window_length / 2) + 1)
        pow = np.zeros(int(window_length / 2) + 1)
        temp = np.zeros(int(window_length / 2) + 1)
        max = np.abs(np.fft.rfft(training_signal[0:0 + window_length] * self.window)) \
            / window_weight
        min = max

        for i in range(0, num_of_windows):
            temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window)) \
                / window_weight
            max = np.maximum(temp, max)
            min = np.minimum(temp, min)
            mean = mean + temp
            pow = pow + np.power(temp, 2)

        mean = mean / num_of_windows
        pow = pow / num_of_windows
        std_dev = np.sqrt(pow - np.power(mean, 2))
        self.mask_top = mean + self.gain * std_dev
        self.mask_bottom = np.maximum(mean - self.gain * std_dev,
                                      np.zeros(int(window_length / 2) + 1))

    def produce(self, X, mask_top=None, mask_bottom=None, window=None):
        """Detects anomalies in telemetry data based on its power spectral density

        Args:
            X: Telemetry data
            mask_top: Spectral mask used to detect anomalies. A new anomaly is detected when a
              frequency component of the telemetry data is above this mask
            mask_bottom: Spectral mask used to detect anomalies. A new anomaly is detected when a
              frequency component of the telemetry data is below this mask.
            window: Window used to calculate the power spectral density of the telemetry data

        Returns:
            anomalies: Data vector consisting of the anomalies detected in the telemetry data

        """

        signal = X

        if window is None:
            if self.window is None:
                return
            window = self.window

        if mask_top is None:
            if self.mask_top is None:
                return
            mask_top = self.mask_top
            mask_bottom = self.mask_bottom

        window_length = len(window)
        anomalies = np.zeros(len(signal))
        window_weight = sum(self.window)
        for i in range(0, len(signal) - window_length - 1):
            sig_freq = np.abs(np.fft.rfft(signal[i:i + window_length] * window)) / window_weight
            anomalies[i] = 0
            for m in range(0, int(window_length / 2) - 1):
                if ((sig_freq[m] > mask_top[m]) or (sig_freq[m] < mask_bottom[m])):
                    anomalies[i] = 1
                    break

        return anomalies
