import numpy as np
import math


def center(X, axis=0):
    return X - np.mean(X, axis=axis)


def normalize(X, axis=0, max_val=None):
    X -= np.min(X, axis=axis)
    if max_val is None:
        X /= np.max(X, axis=axis)
    else:
        X /= max_val
    return X


def standardize(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    X -= mean
    X /= std + 1e-10
    return X
