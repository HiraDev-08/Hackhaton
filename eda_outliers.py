import pandas as pd  # <-- Must import pandas

def iqr_outliers(series):
    """
    Detect outliers using Interquartile Range (IQR) method.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]
