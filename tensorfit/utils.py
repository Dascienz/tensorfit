#!/usr/bin/env python3
import numpy as np
from typing import Tuple


def prepare_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Prepare X and y data. """
    try:
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        assert x.ndim == y.ndim == 1
        assert x.shape == y.shape
        return x, y
    except Exception as e:
        raise ValueError("x and y must be array-like objects: {0}".format(e))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """ Calculate goodness of fit, R^2. """
    rss = np.sum(np.power(y - yhat, 2))
    tss = np.sum(np.power(y - np.mean(y), 2))
    r2 = 1 - (rss / tss)
    return r2
