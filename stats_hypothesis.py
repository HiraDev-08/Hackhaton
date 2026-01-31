from scipy import stats
import numpy as np

def z_test(sample, population_mean):
    """
    Perform one-sample z-test.

    Parameters:
    sample          : list, np.array, pd.Series
    population_mean : float

    Returns:
    float : z-score
    """
    sample = np.array(sample)
    if len(sample) == 0:
        raise ValueError("Sample is empty.")
    std_error = sample.std(ddof=1) / np.sqrt(len(sample))
    if std_error == 0:
        raise ValueError("Standard error is zero, cannot compute z-score.")
    z = (sample.mean() - population_mean) / std_error
    return z

def t_test(sample1, sample2):
    """
    Perform two-sample independent t-test.

    Returns:
    tuple : (t_statistic, p_value)
    """
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Samples cannot be empty.")
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    return t_stat, p_value

def chi_square_test(table):
    """
    Perform chi-square test for independence on a contingency table.

    Parameters:
    table : 2D array-like (rows = categories, columns = outcome counts)

Returns:
    tuple : (chi2_statistic, p_value)
    """
    table = np.array(table)
    if table.ndim != 2:
        raise ValueError("Input table must be 2D.")
    chi2, p, _, _ = stats.chi2_contingency(table)
    return chi2, p
