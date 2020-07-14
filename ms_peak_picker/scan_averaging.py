import math
from collections import defaultdict

import numpy as np


def _binsearch(array, x):
    lo = 0
    hi = len(array)
    while hi != lo:
        mid = (hi + lo) // 2
        y = array[mid]
        err = y - x
        if hi - lo == 1:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


def average_signal(arrays, dx=0.01, weights=None):
    """Average multiple spectras' intensity arrays, with a common m/z axis

    Parameters
    ----------
    arrays : :class:`list` of pairs of :class:`np.ndarray`
        The m/z and intensity arrays to combine
    dx : float, optional
        The m/z resolution to build the averaged m/z axis with
    weights : :class:`list` of :class:`float`, optional
        Weight of each entry in `arrays`. Defaults to 1.0 for each if not provided.

    Returns
    -------
    mz_array: :class:`np.ndarray`
    intensity_array: :class:`np.ndarray`
    """
    if weights is None:
        weights = [1 for _omz in arrays]
    elif len(arrays) != len(weights):
        raise ValueError("`arrays` and `weights` must have the same length")
    try:
        lo = max(min([x.min() for x, y in arrays if len(x)]) - 1, 0)
        hi = max([x.max() for x, y in arrays if len(x)]) + 1
    except ValueError:
        return np.array([]), np.array([])
    arrays = [(x.astype(float), y.astype(float)) for x, y in arrays]
    if isinstance(dx, float):
        mz_array = np.arange(lo, hi, dx)
    elif isinstance(dx, np.ndarray):
        mz_array = dx
    intensity_array = np.zeros_like(mz_array)
    arrays_k = 0
    for mz, inten in arrays:
        weight = weights[arrays_k]
        arrays_k += 1
        contrib = 0
        for i, x in enumerate(mz_array):
            j = _binsearch(mz, x)
            mz_j = mz[j]
            if mz_j < x and j + 1 < mz.shape[0]:
                mz_j1 = mz[j + 1]
                inten_j = inten[j]
                inten_j1 = inten[j + 1]
            elif mz_j > x and j > 0:
                mz_j1 = mz_j
                inten_j1 = inten[j]
                mz_j = mz[j - 1]
                inten_j = mz[j - 1]
            else:
                continue
            contrib = ((inten_j * (mz_j1 - x)) + (inten_j1 * (x - mz_j))) / (mz_j1 - mz_j)
            intensity_array[i] += contrib * weight
    return mz_array, intensity_array / sum(weights)


try:
    _has_c = True
    _average_signal = average_signal
    from ms_peak_picker._c.scan_averaging import average_signal
except ImportError:
    _has_c = False
