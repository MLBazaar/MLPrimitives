import more_itertools as mit
import numpy as np

# Methods to do dynamic error thresholding on timeseries data
# Implementation inspired by: https://arxiv.org/pdf/1802.04431.pdf


def get_forecast_errors(y_hat,
                        y_true,
                        window_size=5,
                        batch_size=30,
                        smoothing_percent=0.05,
                        smoothed=True):
    """
    Calculates the forecasting error for two arrays of data. If smoothed errors desired,
        runs EWMA.
    Args:
        y_hat (list): forecasted values. len(y_hat)==len(y_true).
        y_true (list): true values. len(y_hat)==len(y_true).
        window_size (int):
        batch_size (int):
        smoothing_percent (float):
        smoothed (bool): whether the returned errors should be smoothed with EWMA.
    Returns:
        (list): error residuals. Smoothed if specified by user.
    """
    errors = [abs(y_h - y_t) for y_h, y_t in zip(y_hat, y_true)]

    if not smoothed:
        return errors

    historical_error_window = int(window_size * batch_size * smoothing_percent)
    moving_avg = []
    for i in range(len(errors)):
        left_window = i - historical_error_window
        right_window = i + historical_error_window + 1
        if left_window < 0:
            left_window = 0

        if right_window > len(errors):
            right_window = len(errors)

        moving_avg.append(np.mean(errors[left_window:right_window]))

    return moving_avg


def extract_anomalies(y_true, smoothed_errors, window_size, batch_size, error_buffer):
    """
        Extracts anomalies from the errors.
        Args:
            y_true ():
            smoothed_errors ():
            window_size (int):
            batch_size (int):
            error_buffer (int):
        Returns:
    """
    if len(y_true) <= batch_size * window_size:
        raise ValueError("Window size (%s) larger than y_true (len=%s)."
                         % (batch_size, len(y_true)))

    num_windows = int((len(y_true) - (batch_size * window_size)) / batch_size)

    anomalies_indices = []

    for i in range(num_windows + 1):
        prev_index = i * batch_size
        curr_index = (window_size * batch_size) + (i * batch_size)

        if i == num_windows + 1:
            curr_index = len(y_true)

        window_smoothed_errors = smoothed_errors[prev_index:curr_index]
        window_y_true = y_true[prev_index:curr_index]

        epsilon, sd_threshold = compute_threshold(window_smoothed_errors, error_buffer)

        window_anom_indices = get_anomalies(
            window_smoothed_errors,
            window_y_true,
            sd_threshold,
            i,
            anomalies_indices,
            error_buffer
        )

        # get anomalies from inverse of smoothed errors
        # This was done in the implementation of NASA paper but
        # wasn't referenced in the paper

        # we get the inverse by flipping around the mean
        mu = np.mean(window_smoothed_errors)
        smoothed_errors_inv = [mu + (mu - e) for e in window_smoothed_errors]
        epsilon_inv, sd_inv = compute_threshold(smoothed_errors_inv, error_buffer)
        inv_anom_indices = get_anomalies(
            smoothed_errors_inv,
            window_y_true,
            sd_inv,
            i,
            anomalies_indices,
            len(y_true)
        )

        anomalies_indices = list(set(anomalies_indices + inv_anom_indices))

        anomalies_indices.extend([i_a + i * batch_size for i_a in window_anom_indices])

    # group anomalies
    anomalies_indices = sorted(list(set(anomalies_indices)))
    anomalies_groups = [list(group) for group in mit.consecutive_groups(anomalies_indices)]
    anomaly_sequences = [(g[0], g[-1]) for g in anomalies_groups if not g[0] == g[-1]]

    # generate "scores" for anomalies based on the max distance from epsilon for each sequence
    anomalies_scores = []
    for e_seq in anomaly_sequences:
        denominator = np.mean(smoothed_errors) + np.std(smoothed_errors)
        score = max([
            abs(smoothed_errors[x] - epsilon) / denominator
            for x in range(e_seq[0], e_seq[1])
        ])

        anomalies_scores.append(score)

    return anomaly_sequences, anomalies_scores


def compute_threshold(smoothed_errors, error_buffer, sd_limit=12.0):
    """Helper method for `extract_anomalies` method.
    Calculates the epsilon (threshold) for anomalies.
    """
    mu = np.mean(smoothed_errors)
    sigma = np.std(smoothed_errors)

    max_epsilon = 0
    sd_threshold = sd_limit

    # The treshold is determined dynamically by testing multiple Zs.
    # z is drawn from an ordered set of positive values representing the
    # number of standard deviations above mean(smoothed_errors)

    # here we iterate in increments of 0.5 on the range that the NASA paper found to be good
    for z in np.arange(2.5, sd_limit, 0.5):
        epsilon = mu + (sigma * z)
        below_epsilon, below_indices, above_epsilon = [], [], []

        for i in range(len(smoothed_errors)):
            e = smoothed_errors[i]
            if e < epsilon:
                # save to compute delta mean and delta std
                # these are important for epsilon calculation
                below_epsilon.append(e)
                below_indices.append(i)

            if e > epsilon:
                # above_epsilon values are anomalies
                for j in range(0, error_buffer):
                    if (i + j) not in above_epsilon and (i + j) < len(smoothed_errors):
                        above_epsilon.append(i + j)

                    if (i - j) not in above_epsilon and (i - j) >= 0:
                        above_epsilon.append(i - j)

        if len(above_epsilon) == 0:
            continue

        # generate sequences
        above_epsilon = sorted(list(set(above_epsilon)))
        groups = [list(group) for group in mit.consecutive_groups(above_epsilon)]
        above_sequences = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        mean_perc_decrease = (mu - np.mean(below_epsilon)) / mu
        sd_perc_decrease = (sigma - np.std(below_epsilon)) / sigma
        epsilon = (mean_perc_decrease + sd_perc_decrease) /\
                  (len(above_sequences)**2 + len(above_epsilon))

        # update the largest epsilon we've seen so far
        if epsilon > max_epsilon:
            sd_threshold = z
            max_epsilon = epsilon

    # sd_threshold can be multiplied by sigma to get epsilon
    return max_epsilon, sd_threshold


def get_anomalies(smoothed_errors, y_true, z, window, all_anomalies, error_buffer):
    """
        Helper method to get anomalies.
    """

    mu = np.mean(smoothed_errors)
    sigma = np.std(smoothed_errors)

    epsilon = mu + (z * sigma)

    # compare to epsilon
    errors_seq, anomaly_indices, max_error_below_e = group_consecutive_anomalies(
        smoothed_errors,
        epsilon,
        y_true,
        error_buffer,
        window,
        all_anomalies
    )

    if len(errors_seq) > 0:
        anomaly_indices = prune_anomalies(
            errors_seq,
            smoothed_errors,
            max_error_below_e,
            anomaly_indices
        )

    return anomaly_indices


def group_consecutive_anomalies(smoothed_errors,
                                epsilon,
                                y_true,
                                error_buffer,
                                window,
                                all_anomalies,
                                batch_size=30):
    upper_percentile, lower_percentile = np.percentile(y_true, [95, 5])
    accepted_range = upper_percentile - lower_percentile

    minimum_index = 100  # have a cutoff value for anomalies until model is trained enough

    anomaly_indices = []
    max_error_below_e = 0

    for i in range(len(smoothed_errors)):
        if smoothed_errors[i] <= epsilon or smoothed_errors[i] <= 0.05 * accepted_range:
            # not an anomaly
            continue

        for j in range(error_buffer):
            if (i + j) < len(smoothed_errors) and (i + j) not in anomaly_indices:
                if (i + j) > minimum_index:
                    anomaly_indices.append(i + j)

            if (i - j) < len(smoothed_errors) and (i - j) not in anomaly_indices:
                if (i - j) > minimum_index:
                    anomaly_indices.append(i - j)

    # get all the errors that are below epsilon and which
    #  weren't identified as anomalies to process them
    for i in range(len(smoothed_errors)):
        adjusted_index = i + (window - 1) * batch_size
        if smoothed_errors[i] > max_error_below_e and adjusted_index not in all_anomalies:
            if i not in anomaly_indices:
                max_error_below_e = smoothed_errors[i]

    # group anomalies into continuous sequences
    anomaly_indices = sorted(list(set(anomaly_indices)))
    groups = [list(group) for group in mit.consecutive_groups(anomaly_indices)]
    e_seq = [(g[0], g[-1]) for g in groups if g[0] != g[-1]]

    return e_seq, anomaly_indices, max_error_below_e


def prune_anomalies(e_seq, smoothed_errors, max_error_below_e, anomaly_indices):
    """ Helper method that removes anomalies which don't meet
        a minimum separation from next anomaly.
    """
    # min accepted perc decrease btwn max errors in anomalous sequences
    MIN_PERCENT_DECREASE = 0.05
    e_seq_max, smoothed_errors_max = [], []

    for error_seq in e_seq:
        if len(smoothed_errors[error_seq[0]:error_seq[1]]) > 0:
            sliced_errors = smoothed_errors[error_seq[0]:error_seq[1]]
            e_seq_max.append(max(sliced_errors))
            smoothed_errors_max.append(max(sliced_errors))

    smoothed_errors_max.sort(reverse=True)

    if max_error_below_e > 0:
        smoothed_errors_max.append(max_error_below_e)
    indices_remove = []

    for i in range(len(smoothed_errors_max)):
        if i < len(smoothed_errors_max) - 1:
            delta = smoothed_errors_max[i] - smoothed_errors_max[i + 1]
            perc_change = delta / smoothed_errors_max[i]
            if perc_change < MIN_PERCENT_DECREASE:
                indices_remove.append(e_seq_max.index(smoothed_errors_max[i]))

    for index in sorted(indices_remove, reverse=True):
        del e_seq[index]

    pruned_indices = []

    for i in anomaly_indices:
        for error_seq in e_seq:
            if i >= error_seq[0] and i <= error_seq[1]:
                pruned_indices.append(i)

    return pruned_indices
