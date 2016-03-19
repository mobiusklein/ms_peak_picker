# -*- coding: utf-8 -*-
'''
This module contains a simple translation of Gordon Slyz's implementation of Senko's
Fourier Patterson Charge State determination algorithm.
'''

import numpy as np
from scipy import interpolate


def autocorr(inp):
    """
    Compute the autocorrelation for the input peak sequence

    Parameters
    ----------
    inp : np.ndarray(float64)
        An array representing the sequence of intensities around the peak of interest

    Returns
    -------
    np.ndarray(float64)
        The autocorrelation of the input
    """
    out = np.zeros_like(inp)
    n = len(inp)
    average = inp.mean()

    for i in range(n):
        total = 0.0
        top = n - i - 1
        for j in range(top):
            total += (inp[j] - average) * (inp[j + i] - average)
        if j > 0:
            out[i] = (total / float(n))
        else:
            out[i] = 0.
    return out


def _autocorrelation_scores(peak_index, peak, fwhm, min_mz, max_mz, left_index, right_index):
    """
    Compute an approximation of the fourier transform of the signals around the `peak`. The approximation
    of the fit is performed using the autocorrelation, which is then followed up with cubic spline.

    Parameters
    ----------
    peak_index : PeakIndex
        The signal container
    peak : FittedPeak
        The peak to fit around
    fwhm : float
        The area around the peak to fit
    min_mz : float
        The minimum m/z to search from
    max_mz : float
        The maximum m/z to search to
    left_index : int
        The index of the minimum m/z
    right_index : int
        The index of the maximum m/z

    Raises
    ------
    ValueError
        If not enough points can be found for spline interpolation

    Returns
    -------
    autoscores : np.ndarray(float64)
        The autocorrelation of the signals around `peak`
    num_l: int
        The number of points fitted over

    """
    filtered = peak_index
    sum_of_differences = 0.
    point_count = 0.

    for i in range(len(filtered.mz_array) - 1):
        y1 = filtered.intensity_array[i]
        y2 = filtered.intensity_array[i + 1]
        if y1 > 0 and y2 > 0:
            x1 = filtered.mz_array[i]
            x2 = filtered.mz_array[i + 1]

            sum_of_differences += x2 - x1
            point_count += 1
    if point_count > 5:
        average_differences = sum_of_differences / point_count

        num_l = (np.ceil((max_mz - min_mz) / average_differences))
        num_l = int(num_l + num_l * 1.1)
    else:
        num_points = right_index - left_index + 1
        desired_number = 256
        point_multiplier = np.ceil(desired_number / float(num_points))

        if num_points < 5:
            raise ValueError("Not enough points")

        if num_points < desired_number:
            point_multiplier = max(5, point_multiplier)
            num_l = point_multiplier * num_points
        else:
            num_l = num_points

    interpolant = interpolate.interp1d(filtered.mz_array, filtered.intensity_array, kind='cubic')

    x = np.zeros(num_l)  # [0 for i in range(num_l)]
    y = np.zeros(num_l)

    for i in range(num_l):
        xval = (min_mz + ((max_mz - min_mz) * i) / num_l)
        yval = interpolant(xval)
        x[i] = xval
        y[i] = yval

    autoscores = autocorr(y)
    return autoscores, num_l


def fft_patterson_charge_state(peak_index, peak, max_charge=25):
    """
    Computes the optimal charge state for the given peak using the
    FFT+Patterson Charge State Deconvolution method as published in [1]
    and implemented in Decon2LS.

    A sub-region of the spectrum based upon peak width is selected for
    consideration.

    Parameters
    ----------
    peak_index : PeakIndex
        The PeakIndex which the given Peak is derived from,
        and in which surrounding spectral information will
        be extracted
    peak : FittedPeak
        The peak to compute the optimal charge state for.
    max_charge : int, optional
        The maximal charge state to consider. Defaults to 25.

    Returns
    -------
    charge: positive int
        The optimal charge state for the given peak. In the event
        that the algorithm fails to find an acceptable charge state,
        it will return -1 instead.


    References
    ----------
    [1] Senko, M. W., Beu, S. C., & McLafferty, F. W. (1995).
        Automated assignment of charge states from resolved isotopic peaks for multiply charged ions.
        Journal of the American Society for Mass Spectrometry, 6(1), 52–56.
        http://doi.org/10.1016/1044-0305(94)00091-D

    """
    plus = 1.1
    minus = 0.1
    fwhm = peak.full_width_at_half_max

    left_index = peak_index.get_nearest(peak.mz - fwhm - minus, peak.index)
    right_index = peak_index.get_nearest(peak.mz + fwhm + plus, peak.index)

    filtered = peak_index.slice(left_index, right_index)

    min_mz = filtered.mz_array[0]
    max_mz = filtered.mz_array[len(filtered.mz_array) - 1]

    autoscores, num_l = _autocorrelation_scores(filtered, peak, fwhm, min_mz, max_mz, left_index, right_index)

    start_index = 0
    while (start_index < num_l - 1 and autoscores[start_index] > autoscores[start_index + 1]):
        start_index += 1
    best_autoscore = -1
    best_charge_state = -1

    best_autoscore, best_charge_state = _best_scoring_charge_state(
        min_mz, max_mz, start_index, autoscores, max_charge)

    if best_charge_state == -1:
        return -1

    return_charge_state = -1

    charge_state_list, charge_state_score_map = _generate_charge_state_data(
        min_mz, max_mz, start_index, autoscores, max_charge, best_autoscore)

    for i in range(len(charge_state_list)):
        temp_charge_state = charge_state_list[i]
        skip = False
        for j in range(i):
            if charge_state_list[j] == temp_charge_state:
                skip = True
                break
        if skip:
            continue

        if temp_charge_state > 0:
            another_peak = peak.mz + (1.003 / temp_charge_state)
            if peak_index.has_peak_within_tolerance(another_peak, fwhm):
                return_charge_state = temp_charge_state
                if peak.mz * temp_charge_state < 3000:
                    break
                else:
                    return temp_charge_state
    return return_charge_state


def _best_scoring_charge_state(min_mz, max_mz, start_index, autocorr_scores, max_charge):
    best_autoscore = -1
    best_charge_state = -1

    going_up = False
    was_going_up = False

    npoints = len(autocorr_scores)
    for i in range(start_index, npoints):
        if i < 2:
            continue
        going_up = (autocorr_scores[i] - autocorr_scores[i - 1]) > 0
        if was_going_up and not going_up:
            charge_state = int(npoints / ((max_mz - min_mz) * (i - 1) + 0.5))
            current_score = autocorr_scores[i - 1]
            if abs(current_score / autocorr_scores[0]) > 0.05 and charge_state < max_charge:
                if abs(current_score) > best_autoscore:
                    best_autoscore = abs(current_score)
                    best_charge_state = charge_state
        was_going_up = going_up
    return best_autoscore, best_charge_state


def _generate_charge_state_data(min_mz, max_mz, start_index, autocorr_scores,
                                max_charge, best_autoscore):
    charge_state_list = []
    going_up = False
    was_going_up = False

    npoints = len(autocorr_scores)

    charge_state_score_map = dict()

    for i in range(start_index, npoints):
        if i < 2:
            continue
        going_up = autocorr_scores[i] > autocorr_scores[i - 1]

        if was_going_up and not going_up:
            current_charge_state = (npoints / ((max_mz - min_mz) * (i - 1)))

            current_autocorr_score = autocorr_scores[i - 1]
            if ((current_autocorr_score > best_autoscore * 0.1) and current_charge_state < max_charge):
                charge_state = int(round(current_charge_state))
                charge_state_score_map[charge_state] = current_autocorr_score
                charge_state_list.append(charge_state)

        was_going_up = going_up
    return charge_state_list, charge_state_score_map


try:
    _p_autocorr = autocorr
    from ._c.fft_patterson_charge_state import autocorr, _autocorrelation_scores
except ImportError:
    pass
