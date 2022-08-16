import numpy as np


def gauss(x, mean, variance, scale=2):
    return 1. / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance))


def binsearch(array, value):
    lo = 0
    hi = len(array)
    while hi != lo:
        mid = (hi + lo) / 2
        point = array[mid]
        if value == point:
            return mid
        elif hi - lo == 1:
            return mid
        elif point > value:
            hi = mid
        else:
            lo = mid


def gaussian_smooth(x, y, width=0.02):
    smoothed = np.zeros_like(y)

    n = x.shape[0]
    for i in range(n):
        low_edge = binsearch(x, x[i] - width)
        high_edge = binsearch(x, x[i] + width)
        if high_edge - 1 == low_edge or high_edge == low_edge:
            smoothed[i] = y[i]
            continue
        x_slice = x[low_edge:high_edge]
        y_slice = y[low_edge:high_edge]
        center = np.mean(x_slice)
        spread = np.var(x_slice)
        weights = gauss(x_slice, center, spread)
        val = ((y_slice * weights).sum() / weights.sum())
        if np.isnan(val):
            raise ValueError("NaN")
        smoothed[i] = val
    return smoothed


try:
    _has_c = True
    _gaussian_smooth = gaussian_smooth
    from ms_peak_picker._c.smoother import gaussian_smooth
except ImportError:
    _has_c = False
