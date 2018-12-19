import pandas as pd
import numpy as np


def create_window_sequences(df_timeseries, window_size):
    """
        Function that takes in a Pandas.DataFrame and a window_size then creates
            output arrays that correspond to a timeseries sequence with window_size overlap.
            The output arrays can be fed into a timeseries forecasting model.
        Inputs:
            df_timeseries (Pandas.DataFrame): a Pandas dataframe which has 'timestamp'
                and 'value' columns, and is sorted based on timestamp. 
                The timestamp column is in UNIX format (in seconds).
            window_size (int): number of values that overlap to create the sequence.
        Outputs:
            x (numpy.ndarray): contains the time series sequenced data.
            y (numpy.ndarray): acts as the label for the forecasting problem.
            time (numpy.ndarray): the corresponding timestamps series.
    """
    X = []
    Y = []
    time = []
    for i in range(len(df_timeseries) - window_size):
        X.append(df_timeseries[i: i + window_size]['value'].values.copy().reshape([-1, 1]))
        Y.append(df_timeseries[i + 1: i + window_size + 1]['value'].values.copy().reshape([-1, 1]))
        time.append(df_timeseries.iloc[i + window_size]['timestamp'])
    return np.asarray(X), np.asarray(Y), np.asarray(time)


def aggregate_average_time(df_time_value, interval_time_delta, start_time, end_time):
    """
        Function that aggregates data in a Pandas dataframe by averaging over a given interval. 
            It starts averaging from specified start_time.
        Inputs:
            df_time_value (Pandas.DataFrame): a Pandas dataframe which has 'timestamp' 
                and 'value' columns, and is sorted based on timestamp. The timestamp
                column is in UNIX format (in seconds).
            interval_time_delta (int): an Integer denoting the number of seconds 
                in the desired interval.
            start_time (int): a UNIX time stamp indicating the time to start
                aggregating. Can be smaller than the smallest time stamp value in the dataframe.
            end_time (int): a UNIX time stamp indicating the time to end aggregating. 
                Can be larger than the largest time stamp value in the dataframe.
        Outputs:
            aggregated_df (Pandas.DataFrame): a Pandas dataframe with two colums 
                ('timestamp' and 'value'), where each `timestamp` is the starting time of 
                an interval and the `value` is the result of aggregation. For intervals that 
                don't have data in df_time_value but are still included in start_time 
                and end_time then the value will be NaN.
    """
    start_ts = start_time
    accepted_points = []
    while start_ts < end_time:
        # average the values between start_ts, [start_ts + timedelta (e.g. 6hrs)]
        upper_ts = start_ts + interval_time_delta
        mask = (df_time_value['timestamp'] > start_ts) & (df_time_value['timestamp'] <= upper_ts)
        average_value = df_time_value.loc[mask]['value'].mean(skipna=True)

        accepted_points.append([start_ts, average_value])
        start_ts = upper_ts  # update the timestamp

    new_df = pd.DataFrame(accepted_points, columns=['timestamp', 'value'])
    return new_df
