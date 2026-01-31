import pandas as pd

def missing_percentage(df):
    """
    Calculate percentage of missing values for each column.

    Parameters:
    df : pandas.DataFrame

    Returns:
    pandas.Series : % of missing values per column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    return df.isnull().mean() * 100

def fill_missing(df, method="mean"):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="number").columns:
        if method == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif method == "median":
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif method == "mode":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    return df_copy


    df_copy = df.copy()
    for col in df_copy.select_dtypes(include="number").columns:
        if method == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif method == "median":
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif method == "mode":
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    return df_copy

def drop_missing(df):
    """
    Drop rows with any missing values.

    Parameters:
    df : pandas.DataFrame

    Returns:
    pandas.DataFrame
    """
    return df.dropna().copy()

def duplicate_rows(df):
    """
    Detect duplicate rows in the DataFrame.

    Parameters:
    df : pandas.DataFrame

    Returns:
    pandas.DataFrame : duplicate rows only
    """
    return df[df.duplicated()].copy()

