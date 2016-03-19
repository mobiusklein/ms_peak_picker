cimport cython
cimport numpy as np
from libc cimport math
import numpy as np

from scipy import interpolate

ctypedef np.float64_t DTYPE_t


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=1] autocorr(np.ndarray[DTYPE_t, ndim=1] inp):
    cdef:
        np.ndarray[DTYPE_t, ndim=1] out
        size_t n, j, i, top
        DTYPE_t average, normalizer, total

    out = np.zeros_like(inp)
    average = inp.mean()
    n = len(inp)
    normalizer = <DTYPE_t>n

    for i in range(n):
        total = 0.0
        top = n - i - 1

        for j in range(top):
            total += (inp[j] - average) * (inp[j + i] - average)
            if j > 0:
                out[i] = (total / normalizer)
            else:
                out[i] = 0.
    return out


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef (double, int) _best_scoring_charge_state(double min_mz, double max_mz, size_t start_index, list autocorr_scores, int max_charge):
    cdef:
        double best_autoscore, current_score
        bint going_up, was_going_up
        size_t npoints, i
        int best_charge_state, charge_state
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


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef tuple _autocorrelation_scores(object peak_index, object peak, double fwhm, double min_mz, double max_mz, size_t left_index, size_t right_index):
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

    cdef:
        object filtered
        double sum_of_differences, point_count, y1, y2, x1, x2, num_l_interp, desired_number, num_points
        size_t i
        long num_l
        np.ndarray[DTYPE_t, ndim=1] x, y, autoscores, intensity_array, mz_array
        DTYPE_t xval, yval

    filtered = peak_index
    sum_of_differences = 0.
    point_count = 0.

    mz_array = filtered.mz_array
    intensity_array = filtered.intensity_array

    for i in range(len(mz_array) - 1):
        y1 = intensity_array[i]
        y2 = intensity_array[i + 1]
        if y1 > 0 and y2 > 0:
            x1 = mz_array[i]
            x2 = mz_array[i + 1]

            sum_of_differences += x2 - x1
            point_count += 1
    if point_count > 5:
        average_differences = sum_of_differences / point_count

        num_l_interp = (math.ceil((max_mz - min_mz) / average_differences))
        num_l = int(num_l_interp + num_l_interp * 1.1)
    else:
        num_points = right_index - left_index + 1
        desired_number = 256
        point_multiplier = math.ceil(desired_number / num_points)

        if num_points < 5:
            raise ValueError("Not enough points")

        if num_points < desired_number:
            point_multiplier = max(5, point_multiplier)
            num_l = <long>(point_multiplier * num_points)
        else:
            num_l = <long>num_points

    interpolant = interpolate.interp1d(mz_array, intensity_array, kind='cubic',
                                       copy=False, assume_sorted=True, bounds_error=False)

    x = np.zeros(num_l)  # [0 for i in range(num_l)]
    y = np.zeros(num_l)

    for i in range(num_l):
        xval = (min_mz + ((max_mz - min_mz) * i) / num_l)
        yval = interpolant(xval)
        x[i] = xval
        y[i] = yval

    autoscores = autocorr(y)
    return autoscores, num_l
