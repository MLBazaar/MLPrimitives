def resample(df, rule, time_index, groupby=None, aggregation='mean'):
    """pd.DataFrame.resample adapter.

    Call the `df.resample` method on the given time_index
    and afterwards call the indicated aggregation.

    Optionally group the dataframe by the indicated columns before
    performing the resampling.

    If groupby option is used, the result is a multi-index datagrame.

    Args:
        df (pandas.DataFrame): DataFrame to resample.
        rule (str): The offset string or object representing target conversion.
        groupby (list): Optional list of columns to group by.
        time_index (str): Name of the column to use as the time index.
        aggregation (str): Name of the aggregation function to use.

    Returns:
        pandas.Dataframe: resampled dataframe
    """
    if groupby:
        df = df.groupby(groupby)

    df = df.resample(rule, on=time_index)
    df = getattr(df, aggregation)()
    for column in groupby:
        del df[column]

    return df


def _join_names(names):
    """Join the names of a multi-level index with an underscore."""

    levels = (str(name) for name in names if name != '')
    return '_'.join(levels)


def unstack(df, level=-1, reset_index=True):
    """pd.DataFrame.unstack adapter.

    Call the `df.unstack` method using the indicated level and afterwards
    join the column names using an underscore.

    Args:
        df (pandas.DataFrame): DataFrame to unstack.
        level (str, int or list): Level(s) of index to unstack, can pass level name
        reset_index (bool): Whether to reset the index after unstacking

    Returns:
        pandas.Dataframe: unstacked dataframe
    """
    df = df.unstack(level=level)
    if reset_index:
        df = df.reset_index()
        df.columns = df.columns.map(_join_names)

    return df
