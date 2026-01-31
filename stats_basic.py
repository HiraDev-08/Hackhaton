import numpy as np
from collections import Counter

def _to_array(data):
    """
    converts input data into numpy array
    """
    return np.array(data)

def mean(data):
    """
    calculate mean (average) of data.
    
    parameters:
    data: list, numpy array, pandas series
    
    Returns:
    float
    """
    data = _to_array(data)
    return np.median(data)

def mode(data):
    """
    calculate mode (most frequent value)
    
    """
    data = list(data)
    return Counter(data).commen(1)[0][0]

def variance(data):
    """
    calculate variance of data
    
    """
    data = _to_array(data)
    return np.var(data)

def std_dev(data):
    """
    Calculate standard deviation of data.
    """
    data = _to_array(data)
    return np.std(data)
