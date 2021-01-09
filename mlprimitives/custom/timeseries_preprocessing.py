import warnings

import numpy as np
import pandas as pd


def intervals_to_mask(index, intervals):
    """Create boolean mask from given intervals.

    The function creates an boolean array of same size as the given index array.
    If an index value is within a given interval, the corresponding mask value is `True`.

    Args:
        index (ndarray):
            Array containing the index values.
        intervals (list or ndarray):
            List or array of intervals, consisting of start-index and end-index for each interval.

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


def rolling_window_sequences(X, index, window_size, target_size, step_size, target_column,
                             offset=0, drop=None, drop_windows=False):
    """Create rolling window sequences out of time series data.

    The function creates an array of input sequences and an array of target sequences by rolling
    over the input sequence with a specified window.
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
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        offset (int):
            Indicating the number of steps between the input and the target sequence.
        drop (ndarray or None or str or float or bool):
            Optional. Array of boolean values indicating which values of X are invalid, or value
            indicating which value should be dropped. If not given, `None` is used.
        drop_windows (bool):
            Optional. Indicates whether the dropping functionality should be enabled. If not
            given, `False` is used.

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = X[:, target_column]

    if drop_windows:
        if hasattr(drop, '__len__') and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    max_start = len(X) - window_size - target_size - offset + 1
    while start < max_start:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end + offset:end + offset + target_size])
        X_index.append(index[start])
        y_index.append(index[end + offset])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)


_TIME_SEGMENTS_AVERAGE_DEPRECATION_WARNING = (
    "mlprimitives.custom.timeseries_preprocessing.time_segments_average "
    "is deprecated and will be removed in a future version. Please use "
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate instead."
)


def time_segments_average(X, interval, time_column):
    """Compute average of values over given time span.

    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute average of.
        time_column (int):
            Column of X that contains time values.

    Returns:
        ndarray, ndarray:
            * Sequence of averaged values.
            * Sequence of index values (first index of each averaged segment).
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
    """Aggregate values over given time span.

    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute aggregation of.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.

    Returns:
        ndarray, ndarray:
            * Sequence of aggregated values, one column for each aggregation method.
            * Sequence of index values (first index of each aggregated segment).
    """
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


def cutoff_window_sequences(X, timeseries, window_size, cutoff_time=None, time_index=None):
    """Extract timeseries sequences based on cutoff times.

    Args:
        X (pandas.DataFrame):
            ``pandas.DataFrame`` containing the cutoff time alongside any other values
            that need to be used to filter the matching timeseries data.
            The cutoff time can either be set as the DataFrame index or as a column.
        timeseries (pandas.DataFrame):
            ``pandas.DataFrame`` containing the actual timeseries data. The time index
            and either be set as the DataFrame index or as a column.
        window_size (int, str or Timedelta):
            If an integer is passed, it is the number of elements to take before the
            cutoff time for each sequence. If a string or a Timedelta object is passed,
            it is the period of time we take the elements from.
        cutoff_time (str):
            Optional. If given, the indicated column will be used as the cutoff time.
            Otherwise, the table index will be used.
        time_index (str):
            Optional. If given, the indicated column will be used as the timeseries index.
            Otherwise, the table index will be used.

    Returns:
        numpy.ndarray:
            Numpy array with three dimentions. The frst dimension will have the same
            length as ``X``, and each of the 2D matrices within it will correspond to
            one row in the ``X`` table.
    """

    if cutoff_time:
        X = X.set_index(cutoff_time)

    if time_index:
        timeseries = timeseries.set_index(time_index)

    columns = list(X.columns)

    if not isinstance(window_size, int):
        window_size = pd.to_timedelta(window_size)

    output = list()
    for idx, row in enumerate(X.itertuples()):
        selected = timeseries[timeseries.index < row.Index]

        mask = [True] * len(selected)
        for column in columns:
            mask &= selected.pop(column) == getattr(row, column)

        selected = selected[mask]

        if not isinstance(window_size, int):
            min_time = selected.index[-1] - window_size
            selected = selected.loc[selected.index > min_time]
        else:
            selected = selected.iloc[-window_size:]

        len_selected = len(selected)
        if (len_selected != window_size):
            warnings.warn((
                'Sequence shorter than window_size found: {} < {}. '
                'Output shape is likely to be invalid.'
            ).format(len_selected, window_size))

        output.append(selected.values)

    return np.array(output)
