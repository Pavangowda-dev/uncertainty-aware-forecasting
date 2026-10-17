# calibration.py
import numpy as np

def get_correction(y_cal, lower_cal, upper_cal, alpha=0.1):
    # alpha = 0.1 means we want 90% coverage
    y_cal = np.asarray(y_cal)

    # how far outside the interval was each real value?
    # (a negative number means it was inside)
    scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)

    # pick the right percentile of the scores
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    correction = np.quantile(scores, level, method="higher")
    return correction