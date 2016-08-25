import math
from collections import defaultdict

import numpy as np

from ms_peak_picker import search
from ms_peak_picker.utils import peaklist_to_profile


def average_profile_scans(scan_arrays, width=0.01):
    groups = [[arr.astype(float) for arr in group] for group in scan_arrays]

    mzs = set()
    for mz_array, intensity_array in groups:
        mzs.update(mz_array)

    mzs = sorted(mzs)
    mz_out = []
    intensity_out = []

    _abs = abs
    _len = len

    for mz in mzs:
        cluster_mzs = []
        cluster_intensities = []
        for group in groups:
            mz_array, intensity_array = group
            left_ix = search.get_nearest(mz_array, mz - width, 0)
            left_mz = mz_array[left_ix]
            err = (left_mz - (mz - width))
            abs_err = _abs(err)
            if abs_err > width:
                if err > 0 and left_ix != 0:
                    left_ix -= 1
                elif left_ix != _len(mz_array) - 1:
                    left_ix += 1

            left_mz = mz_array[left_ix]
            err = (left_mz - (mz - width))
            if _abs(err) > (2 * width):
                continue

            right_ix = search.get_nearest(mz_array, mz + width, 0)
            right_mz = mz_array[right_ix]
            err = (right_mz - (mz + width))
            abs_err = _abs(err)
            if abs_err > width:
                if err > 0:
                    right_ix -= 1
                elif right_ix != _len(mz_array) - 1:
                    right_ix += 1

            right_mz = mz_array[right_ix]
            err = (right_mz - (mz + width))
            abs_err = _abs(err)
            if abs_err > (2 * width):
                continue

            mz_values = mz_array[left_ix:(right_ix + 1)]
            intensity_values = intensity_array[left_ix:(right_ix + 1)]

            cluster_mzs.extend(mz_values)
            cluster_intensities.extend(intensity_values)

        cluster_mzs = np.array(cluster_mzs)
        cluster_intensities = np.array(cluster_intensities)
        ix = np.argsort(cluster_mzs)
        cluster_mzs = cluster_mzs[ix]
        cluster_intensities = cluster_intensities[ix]

        u = np.mean(cluster_mzs)
        sd = np.std(cluster_mzs)
        gauss_weights = np.exp(-((cluster_mzs - u) ** 2) / (2 * (sd ** 2)))
        intensity = (gauss_weights * cluster_intensities).sum() / \
            gauss_weights.sum()
        mz_out.append(mz)
        intensity_out.append(intensity)
    return np.array(mz_out, dtype=np.float64), np.array(intensity_out, dtype=np.float64)


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


def average_peak_lists(peak_lists, width=0.01):
    scan_arrays = map(peaklist_to_profile, peak_lists)
    return average_profile_scans(scan_arrays, width=width)
