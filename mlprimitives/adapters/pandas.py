import warnings

from mlprimitives.utils import import_object

_RESAMPLE_AGGS = [
    'mean',
    'median',
    'prod',
    'quantile',
    'std',
    'sum',
    'var',
]


def resample(df, rule, on=None, groupby=(), aggregation='mean',
             reset_index=True, time_index=None):
    """pd.DataFrame.resample adapter.

    Call the `df.resample` method on the given time_index
    and afterwards call the indicated aggregation.

    Optionally group the dataframe by the indicated columns before
    performing the resampling.

    If groupby option is used, the result is a multi-index datagrame.

    Args:
        df (pandas.DataFrame):
            DataFrame to resample.
        rule (str or int):
            The offset string or object representing target conversion or an
            integer value that will be interpreted as the number of seconds.
        on (str or None):
            Name of the column to use as the time index. If ``None`` is given, the
            DataFrame index is used.
        groupby (list):
            Optional list of columns to group by.
        aggregation (callable or str):
            Function or name of the function to use for the aggregation. If a name is given, it
            can either be one of the standard pandas aggregation functions or the fully qualified
            name of a python function that will be imported and used.
        reset_index (bool):
            Whether to reset the index after aggregating
        time_index (str or None):
            Deprecated: This has been renamed to `on`.
            Name of the column to use as the time index. If ``None`` is given, the
            DataFrame is index is used.

    Returns:
        pandas.Dataframe:
            resampled dataframe
    """
    if on is None and time_index is not None:
        message = (
            'resample `time_series` argument deprecated and will be removed'
            ' in future versions of MLPrimitives. Please use `on` instead.'
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        on = time_index

    if groupby:
        df = df.groupby(groupby)

    if isinstance(rule, int):
        rule = '{}s'.format(rule)

    dtir = df.resample(rule, on=on)

    if not callable(aggregation) and aggregation not in _RESAMPLE_AGGS:
        try:
            aggregation = import_object(aggregation)
        except (AttributeError, ImportError, ValueError):
            pass

    df = dtir.aggregate(aggregation)
    for name in df.index.names:
        if name in df:
            del df[name]

    if reset_index:
        df.reset_index(inplace=True)

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
