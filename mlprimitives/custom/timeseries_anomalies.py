"""
Time Series anomaly detection functions.
Implementation inspired by the paper https://arxiv.org/pdf/1802.04431.pdf
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin


def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (array): Ground truth.
        y_hat (array): Predictions array.
        smoothing_window (float): Size of the smoothing window, expressed as a proportion
            of the total length of y.
        smooth (bool): whether the returned errors should be smoothed with EWMA.

    Returns:
        (array): errors
    """
    errors = np.abs(y - y_hat)[:, 0]

    if not smooth:
        return errors

    smoothing_window = int(smoothing_window * len(y))

    return pd.Series(errors).ewm(span=smoothing_window).mean().values


def deltas(errors, epsilon, mean, std):
    """Compute mean and std deltas.

    delta_mean = mean(errors) - mean(all errors below epsilon)
    delta_std = std(errors) - std(all errors below epsilon)
    """
    below = errors[errors <= epsilon]
    if not len(below):
        return 0, 0

    return mean - below.mean(), std - below.std()


def count_above(errors, epsilon):
    """Count number of errors and continuous sequences above epsilon.

    Continuous sequences are counted by shifting and counting the number
    of positions where there was a change and the original value was true,
    which means that a sequence started at that position.
    """
    above = errors > epsilon
    total_above = len(errors[above])

    above = pd.Series(above)
    shift = above.shift(1)
    change = above != shift

    total_consecutive = sum(above & change)

    return total_above, total_consecutive


def z_cost(z, errors, mean, std):
    """Compute how bad a z value is.

    The original formula is::

                 (delta_mean/mean) + (delta_std/std)
        ------------------------------------------------------
        number of errors above + (number of sequences above)^2

    which computes the "goodness" of `z`, meaning that the higher the value
    the better the `z`.

    In this case, we return this value inverted (we make it negative), to convert
    it into a cost function, as later on we will use scipy to minimize it.
    """

    epsilon = mean + z * std

    delta_mean, delta_std = deltas(errors, epsilon, mean, std)
    above, consecutive = count_above(errors, epsilon)

    numerator = -(delta_mean / mean + delta_std / std)
    denominator = above + consecutive ** 2

    if denominator == 0:
        return np.inf

    return numerator / denominator


def find_threshold(errors, z_range=(0, 10)):
    """Find the ideal threshold.

    The ideal threshold is the one that minimizes the z_cost function.
    """

    mean = errors.mean()
    std = errors.std()

    min_z, max_z = z_range
    best_z = min_z
    best_cost = np.inf
    for z in range(min_z, max_z):
        best = fmin(z_cost, z, args=(errors, mean, std), full_output=True, disp=False)
        z, cost = best[0:2]
        if cost < best_cost:
            best_z = z[0]

    return mean + best_z * std


def find_sequences(errors, epsilon):
    """Find sequences of values that are above epsilon.

    This is done following this steps:

        * create a boolean mask that indicates which value are above epsilon.
        * shift this mask by one place, filing the empty gap with a False
        * compare the shifted mask with the original one to see if there are changes.
        * Consider a sequence start any point which was true and has changed
        * Consider a sequence end any point which was false and has changed
    """
    above = pd.Series(errors > epsilon)
    shift = above.shift(1).fillna(False)
    change = above != shift

    index = above.index
    starts = index[above & change].tolist()
    ends = (index[~above & change] - 1).tolist()
    if len(ends) == len(starts) - 1:
        ends.append(len(above) - 1)

    return list(zip(starts, ends))


def find_anomalies(errors, index, z_range=(0, 10)):
    """Find sequences of values that are anomalous.

    We first find the ideal threshold for the set of errors that we have,
    and then find the sequences of values that are above this threshold.

    Lastly, we compute a score proportional to the maximum error in the
    sequence, and finally return the index pairs that correspond to
    each sequence, along with its score.
    """

    threshold = find_threshold(errors, z_range)
    sequences = find_sequences(errors, threshold)

    anomalies = list()
    denominator = errors.mean() + errors.std()
    for start, stop in sequences:
        max_error = errors[start:stop + 1].max()
        score = (max_error - threshold) / denominator
        anomalies.append([index[start], index[stop], score])

    return np.asarray(anomalies)
