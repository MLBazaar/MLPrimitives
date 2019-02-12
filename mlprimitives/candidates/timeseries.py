import numpy as np
import pandas as pd


def rolling_window_sequences(X, window_size, target_size, value_column, time_column):
    """
        Function that takes in a pandas.DataFrame and a window_size then creates
            output arrays that correspond to a timeseries sequence with window_size overlap.
            The output arrays can be fed into a timeseries forecasting model.
            Assumes the input is timeseries sorted.
        Args:
            X (pandas.DataFrame): a pandas dataframe which has 'timestamp'
                and 'value' columns, and is sorted based on timestamp.
                The timestamp column is in UNIX format (in seconds).
            window_size (int): number of values that overlap to create the sequence.
            value_column (string): name of column that has the value field.
            time_column (string): name of column that has the time field.
        Returns:
            (numpy.ndarray): contains the time series sequenced data with each
                entry having window_size rows.
            (numpy.ndarray): acts as the label for the forecasting problem with
                each entry having window_size rows.
            (numpy.ndarray): the corresponding timestamps series.
    """
    output_X = []
    y = []
    time = []
    for start in range(len(X) - window_size - target_size):
        end = start + window_size
        output_X.append(X.iloc[start:end][value_column].values.reshape([-1, 1]))
        y.append(X.iloc[end:end + target_size][value_column].values)
        time.append(X.iloc[end + 1][time_column])

    return np.asarray(output_X), np.asarray(y), np.asarray(time)


def time_segments_average(X, interval, value_column, time_column):
    """
        function that aggregates data in a pandas dataframe by averaging over a given interval.
            it starts averaging from the smallest timestamp in the dataframe and ends at the
            largest timestamp. assumes the input is timeseries sorted.
        args:
            X (pandas.dataframe): a pandas dataframe which has 'timestamp'
                and 'value' columns, and is sorted based on timestamp. the timestamp
                column is in unix format (in seconds).
            interval (int): an integer denoting the number of seconds
                in the desired interval.
            value_column (string): name of column that has the value field.
            time_column (string): name of column that has the time field.
        returns:
            pandas.dataframe: a pandas dataframe with two colums
                ('timestamp' and 'value'), where each `timestamp` is the starting time of
                an interval and the `value` is the result of aggregation.
    """
    start_ts = X[time_column].iloc[0]   # min value
    end_time = X[time_column].iloc[-1]  # max value in dataframe
    accepted_points = []
    while start_ts < end_time:
        # average the values between start_ts, [start_ts + timedelta (e.g. 6hrs)]
        upper_ts = start_ts + interval
        mask = (X[time_column] > start_ts) & (X[time_column] <= upper_ts)
        average_value = X.loc[mask][value_column].mean(skipna=True)

        accepted_points.append([start_ts, average_value])
        start_ts = upper_ts  # update the timestamp

    return pd.DataFrame(accepted_points, columns=[time_column, value_column])
