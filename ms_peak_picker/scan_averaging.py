import math
from collections import defaultdict

import numpy as np

from ms_peak_picker import search


def average_signal(arrays, dx=0.01):
    lo = max(min([x.min() for x, y in arrays]) - 1, 0)
    hi = max([x.max() for x, y in arrays]) + 1
    arrays = [(x.astype(float), y.astype(float)) for x, y in arrays]
    mz_array = np.arange(lo, hi, dx)
    intensity_array = np.zeros_like(mz_array)
    for mz, inten in arrays:
        last = (-1, -1)
        contrib = 0
        for i, x in enumerate(mz_array):
            j = search.get_nearest_binary(mz, x)
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
            if (mz_j, mz_j1) == last:
                # don't update contrib, as the interpolation
                # points haven't changed. If we did, it would cause
                # the terms in the linear interpolation formula which
                # depend upon x to change, leading to a sawtooth pattern
                pass
            else:
                contrib = ((inten_j * (mz_j1 - x)) + (inten_j1 * (x - mz_j))) / (mz_j1 - mz_j)
                last = (mz_j, mz_j1)
            intensity_array[i] += contrib
    return mz_array, intensity_array / len(arrays)


def peak_set_similarity(peak_set_a, peak_set_b, precision=0):
    """Computes the cosine distance between two peak sets, a similarity metric
    ranging between 0 (dissimilar) to 1.0 (similar).

    Parameters
    ----------
    peak_set_a : Iterable of Peak-like
    peak_set_b : Iterable of Peak-like
        The two peak collections to compare. It is usually only useful to
        compare the similarity of peaks of the same class, so the types
        of the elements of `peak_set_a` and `peak_set_b` should match.
    precision : int, optional
        The precision of rounding to use when binning spectra. Defaults to 0

    Returns
    -------
    float
        The similarity between peak_set_a and peak_set_b. Between 0.0 and 1.0
    """
    bin_a = defaultdict(float)
    bin_b = defaultdict(float)

    positions = set()

    for peak in peak_set_a:
        mz = round(peak.mz, precision)
        bin_a[mz] += peak.intensity
        positions.add(mz)

    for peak in peak_set_b:
        mz = round(peak.mz, precision)
        bin_b[mz] += peak.intensity
        positions.add(mz)

    z = 0
    n_a = 0
    n_b = 0

    for mz in positions:
        a = bin_a[mz]
        b = bin_b[mz]
        z += a * b
        n_a += a ** 2
        n_b += b ** 2

    n_ab = math.sqrt(n_a) * math.sqrt(n_b)
    if n_ab == 0.0:
        return 0.0
    else:
        return z / n_ab


try:
    _has_c = True
    _average_signal = average_signal
    from ms_peak_picker._c.peak_statistics import average_signal
except ImportError:
    _has_c = False
