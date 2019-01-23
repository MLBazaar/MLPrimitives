import time
import math
import numpy as np
import matplotlib.pyplot as plt


class Moments:
    def __init__(self, window_length=256, step=1, gain=[1, 1, 1], offset=[0, 0, 0]):
        self.window_length = window_length
        self.step = step
        self.gain = gain
        self.offset = offset

    def fit(self, X):
        self.mean, self.var, self.kur = self.moments(X)
        print('DONE TRAINING')
        print('Mean: ', self.mean)
        print('Variance: ', self.var)
        print('Kurtosis: ', self.kur)

    def produce(self, X, window_length=None, step=None, gain=None, offset=None, fast=True):

        if window_length is None:
            window_length = self.window_length
        else:
            self.window_length = window_length

        if step is None:
            step = self.step
        else:
            self.step = step

        if gain is None:
            gain = self.gain
        else:
            self.gain = gain

        if offset is None:
            offset = self.offset
        else:
            self.offset = offset

        length = len(X)
        anomalies = np.zeros(length)
        if fast:
            for i in range(0, length - window_length, step):
                mean, var, kur = self.moments(X[i:i + window_length - 1])
                if (mean[i] > gain[0] * self.mean + offset[0]) or (mean[i] < self.mean / gain[0] - offset[0]) or (
                        var[i] > gain[1] * self.var + offset[1]) or (
                        kur[i] > gain[2] * self.kur + offset[2]):  # or abs(skew)>gain*abs(self.skew):
                    anomalies[i:i + window_length - 1] = 1
        else:
            mean = np.zeros(length - window_length)
            var = np.zeros(length - window_length)
            kur = np.zeros(length - window_length)

            for i in range(0, length - window_length, step):
                mean[i:i + window_length - 1], var[i:i + window_length - 1], kur[i:i + window_length - 1] = self.moments(
                    X[i:i + window_length - 1])
                if (mean[i] > gain[0] * self.mean + offset[0]) or \
                        (mean[i] < self.mean / gain[0] - offset[0]) or \
                        (var[i] > gain[1] * self.var + offset[1]) or \
                        (kur[i] > gain[2] * self.kur + offset[2]):  # or abs(skew[i])>gain*abs(self.skew):
                    anomalies[i:i + window_length - 1] = 1

            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(mean, label='Mean')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(var, label='Variance')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(kur, label='Kurtosis')
            plt.legend()
        print('DONE DETECTING ANOMALIES - MOMENTS')
        return anomalies

    @staticmethod
    def moments(data):
        length = len(data)
        mean = np.mean(data)
        var = 0
        # skew = 0
        kur = 0
        for i in range(0, length):
            var = var + np.power(data[i] - mean, 2)
            # skew = skew + np.power(data[i] - mean, 3)  # I'm not dividing by the standard deviation
            kur = kur + np.power(data[i] - mean, 4)  # I'm not dividing by the standard deviation
        var = var / length
        # skew = skew / length
        kur = kur / length
        return mean, var, kur


class SpectralMask:

    def __init__(self, method='std_dev', gain=1, window_length=128, beta=4, round_window_length=True):
        self.gain = gain
        self.beta = beta
        self.window_length = window_length
        self.window = None
        self.method = method
        if round_window_length:
            print('Size of the window: ', self.window_length)
            self.window_length = self.next_power_of_2(window_length)
            print('New window size: ', self.window_length)

    def window_design(self, window_length=None, beta=None, show=False):

        if window_length is None:
            window_length = self.window_length
        else:
            self.window_length = window_length

        if beta is None:
            beta = self.beta
        else:
            self.beta = beta

        self.window = np.kaiser(window_length, beta)

        if show:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("Sliding Windows in time")
            plt.plot(self.window)
            plt.xlabel("time [sec]")
            plt.ylabel("amplitude")
            plt.subplot(2, 1, 2)
            plt.title("Sliding Windows in frequency")
            plt.plot(np.fft.fftshift(np.abs(np.fft.fft(self.window))))
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude")
            plt.show(block=False)

        return self.window

    def fit(self, X, method=None, gain=None, window_length=None, beta=None, show=False):

        start = time.time()
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

        self.window_design(window_length, beta, show)

        if method == 'std_dev':
            self.fit_freq_min_max(training_signal, show)
        else:
            self.fit_freq_std_dev(training_signal, show)

        self.time_fit = time.time() - start
        print('DONE TRAINING')
        print('Time consumed training: ', self.time_fit)

    def fit_freq_min_max(self, training_signal, show):
        window_length = len(self.window)
        window_weight = sum(self.window) / 2
        max_mask = np.zeros(int(window_length / 2) + 1)
        min_mask = np.zeros(int(window_length / 2) + 1)

        for i in range(0, len(training_signal) - window_length - 1):
            temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window)) / window_weight
            max_mask = np.maximum(max_mask, temp)
            min_mask = np.minimum(min_mask, temp)

        self.mask_top = self.gain * max_mask
        self.mask_bottom = min_mask / self.gain

        if show:
            plt.figure()
            plt.title("Frequency Mask - fit_freq_max_min")
            plt.plot(self.mask_top)
            plt.plot(self.mask_bottom)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude")
            plt.show(block=False)

    def fit_freq_std_dev(self, training_signal, show):
        window_length = len(self.window)
        window_weight = sum(self.window) / 2
        num_of_windows = len(training_signal) - window_length - 1
        mask = np.zeros(window_length)

        mean = np.zeros(int(window_length / 2) + 1)
        pow = np.zeros(int(window_length / 2) + 1)
        temp = np.zeros(int(window_length / 2) + 1)
        max = np.abs(np.fft.rfft(training_signal[0:0 + window_length] * self.window)) / window_weight
        min = max

        for i in range(0, num_of_windows):
            temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window)) / window_weight
            max = np.maximum(temp, max)
            min = np.minimum(temp, min)
            mean = mean + temp
            pow = pow + np.power(temp, 2)
        mean = mean / num_of_windows
        pow = pow / num_of_windows
        std_dev = np.sqrt(pow - np.power(mean, 2))
        self.mask_top = mean + self.gain * std_dev
        self.mask_bottom = np.maximum(mean - self.gain * std_dev, np.zeros(int(window_length / 2) + 1))

        # scipy -> same performance as numpy

        # Classical approach - too slow
        # mean = zeros(int(window_length/2)+1)
        # pow = zeros(int(window_length/2)+1)
        # temp = zeros(int(window_length/2) + 1)
        # for i in range(0,num_of_windows):
        #     temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window))
        #     mean = mean + temp
        # mean = mean / num_of_windows
        # for i in range(0, num_of_windows):
        #     temp = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window))
        #     pow = pow + np.power(temp-mean, 2)
        # mask = mean + gain * np.sqrt(pow/num_of_windows)

        # Approach using multi-cores
        # mean = zeros(int(window_length/2) + 1)
        # pow = zeros(int(window_length/2) + 1)
        # temp = zeros(int(window_length/2) + 1)
        # start = time.time()
        # num_cores = multiprocessing.cpu_count()
        # print('Number of cores: ', num_cores)
        # temp = Parallel(n_jobs=num_cores) /
        #        (delayed(self.par_fft)(training_signal[i:i+L] * self.window) for i in range(num_of_windows))
        # mean = sum(mean)/num_of_windows
        # pow = (temp-mean)*(temp-mean)

        # Approach using Vectors - too much memory
        # mean = zeros(int(window_length/2)+1)
        # pow = zeros(int(window_length/2)+1)
        # temp = zeros(int(window_length/2) + 1)
        # temp = zeros((num_of_windows,window_length))
        # for i in range(0, num_of_windows):
        #     temp[i] = np.abs(np.fft.rfft(training_signal[i:i + window_length] * self.window))
        # mean = sum(temp)/num_of_windows
        # pow = (temp-mean)*(temp-mean)

        self.time_fit_freq_std_dev_mask = time.time() - start

        if show:
            plt.figure()
            plt.title("Frequency Mask - fit_freq_std_dev")
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, self.mask_top, label='StdDev Mask - Top')
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, max, label='Max Mask')
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, mean, label='Mean')
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, min, label='Min Mask')
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, self.mask_bottom, label='StdDev Mask - Bottom')
            plt.xlabel("Normalized frequency - f/fs [cycles/sample]")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.show(block=False)

            plt.figure()
            plt.title("Confidence Intervals for Frequency Mask - fit_freq_std_dev")
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, mean)
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, mean + 3 * std_dev / np.sqrt(num_of_windows))
            plt.plot(np.arange(0, window_length / 2 + 1) / window_length, mean - 3 * std_dev / np.sqrt(num_of_windows))
            plt.xlabel("Normalized frequency - f/fs [cycles/sample]")
            plt.ylabel("Amplitude")
            plt.show(block=False)

    def produce(self, X, mask_top=None, mask_bottom=None, window=None):

        start = time.time()
        signal = X
        if window is None:
            if self.window is None:
                print("freqCompare: Window not defined yet")
                return
            window = self.window
        if mask_top is None:
            if self.mask_top is None:
                print("freqCompare: std_dev_mask not defined yet")
                return
            mask_top = self.mask_top
            mask_bottom = self.mask_bottom

        window_length = len(window)
        anomalies = np.zeros(len(signal))
        for i in range(0, len(signal) - window_length - 1):
            sig_freq = np.abs(np.fft.rfft(signal[i:i + window_length] * window)) / window_length
            anomalies[i] = 0
            for m in range(0, int(window_length / 2) - 1):
                if ((sig_freq[m] > mask_top[m]) or (sig_freq[m] < mask_bottom[m])):
                    anomalies[i] = 1
                    break
        self.time_produce = time.time() - start
        print('DONE DETECTING ANOMALIES - SPECTRAL MASK')
        print('Time consumed detecting anomalies: ', self.time_produce)

        return anomalies  # , self.time_freq_compare

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else 2 ** math.ceil(math.log2(x))

    @staticmethod
    def par_fft(data):
        return np.abs(np.fft.rfft(data))

