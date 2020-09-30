#!/usr/bin/env python3
import numpy as np


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """ Calculate goodness of fit, R^2. """
    rss = np.sum(np.power(y - yhat, 2))
    tss = np.sum(np.power(y - np.mean(y), 2))
    r2 = 1 - (rss / tss)
    return r2
