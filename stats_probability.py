import numpy as np
from scipy.stats import norm

def z_score(x, mean, std):
     """
     calculate z_score for given value(s).
     Parameters:
    x    : float or array-like, value(s) to standardize
    mean : float, mean of the distribution
    std  : float, standard deviation of the distribution

    Returns:
    float or np.array : z-score(s)
 
     """
     x = np.array(x)
     if std == 0:
         raise ValueError("standard daviation cannot be zero")
     return (x-mean)/ std
def normal_pdf(x,mean,std):
 
    x = np.array(x)
    return norm.pdf(x,mean,std)

