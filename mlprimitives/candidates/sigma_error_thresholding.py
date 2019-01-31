import more_itertools as mit
import numpy as np

# Methods to do outlier detection based on sigma thresholds


def extract_anomalies(errors, num_sigmas):
    std = np.std(np.array(errors))
    anomalies_indices = [i for i in range(len(errors))
                         if errors[i] >= (num_sigmas * std)]

    anomalies_indices = sorted(list(set(anomalies_indices)))
    group_anomalies = [list(group) for group in mit.consecutive_groups(anomalies_indices)]
    anomaly_sequences = [(g[0], g[-1]) for g in group_anomalies if g[0] != g[-1]]

    return anomaly_sequences
