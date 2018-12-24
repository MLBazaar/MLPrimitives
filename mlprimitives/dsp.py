import math
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import *
import time

class freqAnalysis:

    def __init__(self, gain = 1, windowSize = 129, beta = 4):
        self.gain = gain
        self.beta = beta
        self.windowSize = windowSize
        #self.gain = gain
        #self.mask = mask

    def windowDesign(self, windowSize=None, beta=None, show=False):
        if windowSize==None:
            windowSize = self.windowSize
        else:
            self.windowSize = windowSize

        if beta==None:
            beta = self.beta
        else:
            self.beta = beta

        self.window = np.kaiser(windowSize, beta)

        if show:
            plt.figure()
            plt.subplot(2,1,1)
            plt.title("Sliding Windows in time")
            plt.plot(self.window)
            plt.xlabel("time [sec]")
            plt.ylabel("amplitude")
            plt.subplot(2,1,2)
            plt.title("Sliding Windows in frequency")
            plt.plot(np.fft.fftshift(np.abs(np.fft.fft(self.window))))
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude")
            plt.show()

        return self.window

    def fitFreqMax(self, trainingSignal, gain=None, windowSize=None, beta=None, show=False):
        if gain == None:
            gain = self.gain
        else:
            self.gain = gain
        if windowSize==None:
            windowSize = self.windowSize
        else:
            self.windowSize = windowSize
        if beta == None:
            beta = self.beta
        else:
            self.beta = beta
        start = time.time()
        self.windowDesign(windowSize, beta, show)
        L = len(self.window)
        mask = zeros(L)
        for i in range(0,len(trainingSignal)-L-1):
            temp = np.abs(np.fft.fft(trainingSignal[i:i+L]*self.window))
            mask = maximum(mask, temp)
        self.MaxMask = gain*mask
        self.xTimeFitFreqMax = time.time() - start
        if show:
            plt.figure()
            plt.title("Frequency Mask - fitFreqMax")
            plt.plot(np.fft.fftshift(self.MaxMask))
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude")
            plt.show()
        return self.MaxMask #, self.xTimeFitFreqMax

    def fitFreqStdDev(self, X):
        trainingSignal = X
        gain = None
        windowSize = None
        beta = None
        show = False
        if gain == None:
            gain = self.gain
        else:
            self.gain = gain
        if windowSize==None:
            windowSize = self.windowSize
        else:
            self.windowSize = windowSize
        if beta == None:
            beta = self.beta
        else:
            self.beta = beta
        start = time.time()
        self.windowDesign(windowSize, beta, show)
        L = len(self.window)
        M = len(trainingSignal)-L-1
        mask = zeros(L)
        mean = zeros(L)
        pow =  zeros(L)
        for i in range(0,M):
            temp = np.abs(np.fft.fft(trainingSignal[i:i+L]*self.window))
            mean = mean + temp
        mean = mean/M
        for i in range(0, M):
            temp = np.abs(np.fft.fft(trainingSignal[i:i + L] * self.window))
            pow = pow + np.power(temp-mean, 2)
        self.StdDevMask = mean + gain*np.sqrt(pow/(M-1))
        self.xTimeFitFreqStdDev = time.time() - start
        if show:
            plt.figure()
            plt.title("Frequency Mask - fitFreqStdDev")
            plt.plot(np.fft.fftshift(self.StdDevMask))
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude")
            plt.show()
        #return self.StdDevMask #, self.xTimeFitFreqStdDev

    def produceFreqCompare(self, X):
        signal = X
        mask = []
        window = []
        method = 'StdDev'
        start = time.time()
        if window == []:
            if self.window == []:
                print("freqCompare: Window not defined yet")
                return
            window = self.window
        if mask == []:
            if method == 'StdDev':
                if self.StdDevMask == []:
                    print("freqCompare: StdDevMask not defined yet")
                    return
                mask = self.StdDevMask
            else:
                if self.MaxMask == []:
                    print("freqCompare: MaxMask not defined yet")
                    return
                mask = self.MaxMask
        L = len(window)
        anomalies = zeros(len(signal))
        for i in range(0, len(signal) - L - 1):
            sigFreq = np.abs(np.fft.fft(signal[i:i + L] * window))
            anomalies[i] = 0
            for m in range(0, L - 1):
                if sigFreq[m] > mask[m]:
                    anomalies[i] = 1
                    break
        self.xTimeFreqCompare = time.time() - start
        return anomalies #, self.xTimeFreqCompare