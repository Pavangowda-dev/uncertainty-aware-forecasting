# metrics.py
import numpy as np

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def coverage(actual, lower, upper):
    # what fraction of real values fall inside the interval?
    inside = (actual >= lower) & (actual <= upper)
    return np.mean(inside)

def avg_width(lower, upper):
    # how wide is the interval on average? (narrower is better, if coverage is OK)
    return np.mean(upper - lower)    