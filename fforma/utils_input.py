from typing import List, Union, Optional
import pandas as pd

def _check_valid_df(df: Union[pd.DataFrame, pd.Series]) -> bool:
    """Check if input is a valid DataFrame or Series.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input data to validate

    Returns
    -------
    bool
        True if DataFrame, False if Series
    """
    is_pandas_df = isinstance(df, pd.DataFrame)
    assert  is_pandas_df or isinstance(df, pd.Series)

    return is_pandas_df

def _check_valid_columns(df: Union[pd.DataFrame, pd.Series],
                         cols: Optional[List[str]] = None,
                         cols_index: Optional[List[str]] = None) -> None:
    """Check if DataFrame has required columns.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        Input data to validate
    cols : Optional[List[str]]
        Required column names (default: ['unique_id','ds', 'y'])
    cols_index : Optional[List[str]]
        Required index column names (default: ['unique_id', 'ds'])
    """
    if cols is None:
        cols = ['unique_id','ds', 'y']
    if cols_index is None:
        cols_index = ['unique_id', 'ds']

    correct_cols_df = all([item in df.columns for item in cols])
    correct_cols_index = all([item in df.index.names for item in cols_index])

    assert correct_cols_df or correct_cols_index

def _check_same_type(df_x: Union[pd.DataFrame, pd.Series], df_y: Union[pd.DataFrame, pd.Series]) -> None:
    """Check if two DataFrames/Series are of the same type.

    Parameters
    ----------
    df_x : Union[pd.DataFrame, pd.Series]
        First data object
    df_y : Union[pd.DataFrame, pd.Series]
        Second data object
    """
    assert type(df_x) == type(df_y)

def _check_passed_dfs(df_x: Union[pd.DataFrame, pd.Series], df_y: Union[pd.DataFrame, pd.Series]) -> bool:
    """Check if both passed DataFrames are valid and of same type.

    Parameters
    ----------
    df_x : Union[pd.DataFrame, pd.Series]
        First data object to validate
    df_y : Union[pd.DataFrame, pd.Series]
        Second data object to validate

    Returns
    -------
    bool
        True if both are DataFrames, False if both are Series
    """
    for df in [df_x, df_y]:
        is_pandas_df = _check_valid_df(df)
        _check_valid_columns(df)

    _check_same_type(df_x, df_y)

    return is_pandas_df
