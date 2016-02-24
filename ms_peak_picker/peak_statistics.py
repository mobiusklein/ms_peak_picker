import numpy as np

from .utils import range
from .search import get_nearest


def find_signal_to_noise(target_val, intensity_array, index):
    min_intensity_left = 0
    min_intensity_right = 0
    size = len(intensity_array) - 1
    if target_val == 0:
        return 0
    if index <= 0 or index >= size:
        return 0

    for i in range(index, 0, -1):
        if intensity_array[i + 1] >= intensity_array[i] and intensity_array[i - 1] > intensity_array[i]:
            min_intensity_left = intensity_array[i]
            break
    else:
        min_intensity_left = intensity_array[0]

    for i in range(index, size):
        if intensity_array[i + 1] >= intensity_array[i] and intensity_array[i - 1] > intensity_array[i]:
            min_intensity_right = intensity_array[i]
            break
    else:
        min_intensity_right = intensity_array[size]

    if min_intensity_left == 0:
        if min_intensity_right == 0:
            return 100
        else:
            return target_val / min_intensity_right

    if min_intensity_right < min_intensity_left and min_intensity_right != 0:
        return target_val / min_intensity_right
    return target_val / min_intensity_left


def quadratic_fit(mz_array, intensity_array, index):
    if index < 1:
        return mz_array[0]
    elif index > len(mz_array) - 1:
        return mz_array[-1]
    x1, x2, x3 = mz_array[(index - 1):(index + 2)]
    y1, y2, y3 = intensity_array[(index - 1):(index + 2)]

    d = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
    if d == 0:  # If the interpolated intensity is 0, the peak fitting is no better than the peak
        return x2
    mz_fit = ((x1 + x2) - ((y2 - y1) * (x3 - x2) * (x1 - x3)) / d) / 2.0
    return mz_fit


def curve_reg(x, y, n, terms, nterms):
    """
    Fit a linear least squares regression

    Parameters
    ----------
    x : array
    y : array
    n : int
    terms : array
        Mutated to pass back coefficients
        of fit.
    nterms : int
        Number of terms

    Returns
    -------
    float
    """
    weights = np.ones(n)

    # Weighted powers of x transposed
    At = np.zeros((nterms + 1, n))
    for i in range(n):
        At[0, i] = weights[i]
        for j in range(1, nterms + 1):
            At[j, i] = At[j - 1, i] * x[i]

    Z = np.zeros((n, 1))
    for i in range(n):
        Z[i, 0] = weights[i] * y[i]

    # ((AtA)^-1)At
    At_T = At.T
    At_At_T = At.dot(At_T)
    I_At_At_T = np.linalg.inv(At_At_T)
    At_Ai_At = I_At_At_T.dot(At)

    B = At_Ai_At.dot(Z)

    mse = 0
    out = np.zeros((2, n))
    for i in range(n):
        terms[0] = B[0, 0]
        yfit = B[0, 0]
        xpow = x[i]
        for j in range(1, nterms):
            terms[j] = B[j, 0]
            yfit += B[j, 0] * xpow
            xpow = xpow * x[i]
        out[0, i] = yfit
        out[1, i] = y[i] - yfit
        mse += y[i] - yfit
    return mse


def find_full_width_at_half_max(mz_array, intensity_array, data_index, signal_to_noise=0.):
    points = 0
    peak = intensity_array[data_index]
    peak_half = peak / 2.
    mass = mz_array[data_index]

    coef = np.zeros(2)

    if peak == 0.0:
        return 0.

    size = len(mz_array) - 1
    if data_index <= 0 or data_index >= size:
        return 0.

    upper = mz_array[0]
    for index in range(data_index, -1, -1):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if ((Y1 < peak_half) or (np.fabs(mass - current_mass) > 5.0) or (
                (index < 1 or intensity_array[index - 1] > Y1) and (
                    index < 2 or intensity_array[index - 2] > Y1) and (signal_to_noise < 4.0))):
            Y2 = intensity_array[index + 1]
            X1 = mz_array[index]
            X2 = mz_array[index + 1]

            if ((Y2 - Y1 != 0) and (Y1 < peak_half)):
                upper = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                upper = X1
                points = data_index - index + 1
                if points >= 3:

                    vect_mzs = []
                    vect_intensity = []

                    for j in range(points - 1, -1, -1):
                        vect_mzs.append(mz_array[data_index - j])
                        vect_intensity.append(intensity_array[data_index - j])

                    j = 0
                    while j < points and (vect_intensity[0] == vect_intensity[j]):
                        j += 1

                    if j == points:
                        return 0.

                    # coef will contain the results
                    curve_reg(vect_intensity, vect_mzs, points, coef, 1)
                    upper = coef[1] * peak_half + coef[0]
            break

    lower = mz_array[size]
    for index in range(size):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if((Y1 < peak_half) or (np.fabs(mass - current_mass) > 5.0) or (
                (index > size - 1 or intensity_array[index + 1] > Y1) and (
                index > size - 2 or intensity_array[index + 2] > Y1) and signal_to_noise < 4.0)):
            Y2 = intensity_array[index - 1]
            X1 = mz_array[index]
            X2 = mz_array[index - 1]

            if((Y2 - Y1 != 0) and (Y1 < peak_half)):
                lower = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                lower = X1
                points = index - data_index + 1
                if points >= 3:
                    vect_mzs = []
                    vect_intensity = []

                    for k in range(points - 1, -1, -1):
                        vect_mzs.append(mz_array[index - k])
                        vect_intensity.append(intensity_array[index - k])
                    j = 0
                    while (j < points) and (vect_intensity[0] == vect_intensity[j]):
                        j += 1

                    if j == points:
                        return 0.0

                    # coef will contain the result
                    curve_reg(vect_intensity, vect_mzs, points, coef, 1)
                    lower = coef[1] * peak_half + coef[0]
            break

    if upper == 0.0:
        return 2 * np.fabs(mass - lower)
    if lower == 0.0:
        return 2 * np.fabs(mass - upper)
    return np.fabs(upper - lower)


def lorenztian_least_squares(mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop):
    root_mean_squared_error = 0

    for index in range(lstart, lstop + 1):
        u = 2 / float(full_width_at_half_max) * (mz_array[index] - vo)
        Y1 = int(amplitude / float(1 + u * u))
        Y2 = intensity_array[index]

        root_mean_squared_error += (Y1 - Y2) * (Y1 - Y2)

    return root_mean_squared_error


def lorenztian_fit(mz_array, intensity_array, index, full_width_at_half_max):
    amplitude = intensity_array[index]
    vo = mz_array[index]
    E = np.fabs((vo - mz_array[index + 1]) / 100.0)

    if index < 1:
        return mz_array[index]
    elif index >= len(mz_array) - 1:
        return mz_array[-1]

    lstart = get_nearest(mz_array, vo + full_width_at_half_max, index) + 1
    lstop = get_nearest(mz_array, vo - full_width_at_half_max, index) - 1

    CE = lorenztian_least_squares(mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
    for i in range(50):
        le = CE
        vo = vo + E
        CE = lorenztian_least_squares(mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        if (CE > le):
            break

    vo = vo - E
    CE = lorenztian_least_squares(mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
    for i in range(50):
        le = CE
        vo = vo - E
        CE = lorenztian_least_squares(mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        if (CE > le):
            break

    vo += E
    return vo

try:
    from . import _peak_statistics
    _find_signal_to_noise = find_signal_to_noise
    # _find_full_width_at_half_max = find_full_width_at_half_max
    # _curve_reg = curve_reg

    find_signal_to_noise = _peak_statistics.find_signal_to_noise
    # find_full_width_at_half_max = _peak_statistics.find_full_width_at_half_max
    # curve_reg = _peak_statistics.curve_reg
except ImportError:
    pass
