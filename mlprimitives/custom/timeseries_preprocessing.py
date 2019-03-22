import warnings

import numpy as np
import pandas as pd


def rolling_window_sequences(X, index, window_size, target_size, target_column):
    """Create rolling window sequences out of timeseries data."""
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()

    target = X[:, target_column]

    for start in range(len(X) - window_size - target_size + 1):
        end = start + window_size
        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING = (
    "mlprimitives.custom.timeseries_preprocessing.time_segments_average "
    "is deprecated and will be removed in a future version. Please use "
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate instead."
)


def time_segments_average(X, interval, time_column):
    """Compute average of values over fixed length time segments."""
    warnings.warn(_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING, DeprecationWarning)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        means = subset.mean(skipna=True).values
        values.append(means)
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)


def time_segments_aggregate(X, interval, time_column, method=['mean']):
    """Aggregate values over fixed length time segments."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)
