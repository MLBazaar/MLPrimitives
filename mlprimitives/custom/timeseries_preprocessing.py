import warnings

import numpy as np
import pandas as pd


def intervals_to_mask(index, intervals):
    """Create boolean mask from given intervals.

    We create an array of same size as the given index array containing boolean values.
    If an index value is within a given interval, the corresponding mask value is `True`.

    Args:
        index (ndarray):
            Array containing the index values.
        intervals (list):
            List of intervals, consisting of start-index and end-index for each interval

    Returns:
        ndarray:
            Array of boolean values, with one boolean value for each index value (`True` if the
            index value is contained in a given interval, otherwise `False`).
    """
    mask = list()

    if intervals is None or len(intervals) == 0:
        return np.zeros((len(index), ), dtype=bool)

    for value in index:
        for start, end in intervals:
            if start <= value <= end:
                mask.append(True)
                break
        else:
            mask.append(False)

    return np.array(mask)


def rolling_window_sequences(X, index, window_size, target_size, target_column, drop,
                             drop_windows=False):
    """Create rolling window sequences out of timeseries data.

    Create an array of input sequences and an array of target sequences by rolling over
    the input sequence with a window.
    Optionally, certain values can be dropped from the sequences.

    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        target_column (int):
            Indicating which column of X is the target.
        drop (ndarray or None or str or float or bool):
            Array of boolean values indicating which values of X are invalid, or value
            indicating which value should be dropped.
        drop_windows (bool):
            Optional. Indicates whether the dropping functionality should be enabled. If not
            given, `False` is used.

    Returns:
        ndarray:
            Array containing the input sequences.
        ndarray:
            Array containing the target sequences.
        ndarray:
            Array containing the index values of the input sequences.
        ndarray:
            Array containing the index values of the target sequences.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = X[:, target_column]

    if drop_windows:
        if hasattr(drop, '__len__'):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')

        else:
            drop = X == drop

    start = 0
    while start < len(X) - window_size - target_size + 1:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + 1

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING = (
    "mlprimitives.custom.timeseries_preprocessing.time_segments_average "
    "is deprecated and will be removed in a future version. Please use "
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate instead."
)


def time_segments_average(X, interval, time_column):
    """Compute average of values over fixed length time segments.

    Args:
        X (ndarray):
            N-dimensional sequence of values.
        interval (int):
            Length of segments to compute average of.
        time_column (int):
            Column of X that contains time values.

    Returns:
        ndarray:
            Sequence of averaged values.
        ndarray:
            Sequence of index values (first index of each averaged segment).
    """
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
    """Aggregate values over fixed length time segments.

    Args:
        X (ndarray):
            N-dimensional sequence of values.
        interval (int):
            Length of segments to aggregate.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.

    Returns:
        ndarray:
            Sequence of aggregated values, one column for each aggregation method.
        ndarray:
            Sequence of index values (first index of each aggregated segment).
    """
    print(method)
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
