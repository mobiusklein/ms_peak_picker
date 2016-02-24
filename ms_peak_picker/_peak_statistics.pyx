cimport cython
cimport numpy as np
from libc cimport math
import numpy as np

ctypedef np.float64_t DTYPE_t

cdef bint isclose(DTYPE_t x, DTYPE_t y, DTYPE_t rtol=1.e-5, DTYPE_t atol=1.e-8):
    return abs(x-y) <= (atol + rtol * abs(y))

cdef bint aboutzero(DTYPE_t x):
    return isclose(x, 0)


@cython.boundscheck(False)
cpdef DTYPE_t find_signal_to_noise(double target_val, np.ndarray[DTYPE_t, ndim=1] intensity_array, size_t index):
    cdef:
        DTYPE_t min_intensity_left, min_intensity_right
        size_t size, i

    min_intensity_left = 0
    min_intensity_right = 0
    size = intensity_array.shape[0] - 1
    if aboutzero(target_val):
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

    if aboutzero(min_intensity_left):
        if aboutzero(min_intensity_right):
            return 100
        else:
            return target_val / min_intensity_right

    if min_intensity_right < min_intensity_left and not aboutzero(min_intensity_right):
        return target_val / min_intensity_right
    return target_val / min_intensity_left


@cython.boundscheck(False)
cpdef DTYPE_t curve_reg(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, size_t n, np.ndarray[DTYPE_t, ndim=1] terms, size_t nterms):
    cdef:
        np.ndarray[DTYPE_t, ndim=1] weights
        np.ndarray[DTYPE_t, ndim=2] At, Z, At_T, At_At_T, I_At_At_T, At_Ai_At
        np.ndarray[DTYPE_t, ndim=2] B, out
        DTYPE_t mse, yfit, xpow
        size_t i, j

    weights = np.ones(n)
    
    # Weighted powers of x transposed
    # Like Vandermonte Matrix?
    At = np.zeros((nterms + 1, n))
    for i in range(n):
        At[0, i] = weights[i]
        for j in range(1, nterms + 1):
            At[j, i] = At[j - 1, i] * x[i]
    
    Z = np.zeros((n, 1))
    for i in range(n):
        Z[i, 0] = weights[i] * y[i]
    
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
        mse += y[i]-yfit
    return mse


@cython.boundscheck(False)
cpdef DTYPE_t find_full_width_at_half_max(np.ndarray[DTYPE_t, ndim=1] mz_array, np.ndarray[DTYPE_t, ndim=1] intensity_array, size_t data_index,
                                          double signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse
        DTYPE_t upper, lower
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1] coef
        list vect_mzs, vect_intensity

    points = 0
    peak = intensity_array[data_index]
    peak_half = peak / 2.
    mass = mz_array[data_index]
    
    coef = np.zeros(2)
    
    if aboutzero(peak):
        return 0.

    size = len(mz_array) - 1
    if data_index <= 0 or data_index >= size:
        return 0.
    
    upper = mz_array[0]
    for index in range(data_index, -1, -1):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if ((Y1 < peak_half) or (math.fabs(mass - current_mass) > 5.0) or (
                (index < 1 or intensity_array[index - 1] > Y1) and (
                 index < 2 or intensity_array[index - 2] > Y1) and (signal_to_noise < 4.0))):
            Y2 = intensity_array[index + 1]
            X1 = mz_array[index]
            X2 = mz_array[index + 1]

            if (not aboutzero(Y2 - Y1) and (Y1 < peak_half)):
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
                    mse = curve_reg(vect_intensity, vect_mzs, points, coef, 1)
                    upper = coef[1] * peak_half + coef[0]
            break
    lower = mz_array[size]
    for index in range(size):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if((Y1 < peak_half) or (math.fabs(mass - current_mass) > 5.0) or ((index > size - 1 or intensity_array[index + 1] > Y1) and (
                    index > size - 2 or intensity_array[index + 2] > Y1) and signal_to_noise < 4.0)):
            Y2 = intensity_array[index - 1]
            X1 = mz_array[index]
            X2 = mz_array[index - 1]
            
            if(not aboutzero(Y2 - Y1) and (Y1 < peak_half)):
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
                    mse = curve_reg(vect_intensity, vect_mzs, points, coef, 1)
                    lower = coef[1] * peak_half + coef[0]
            break

    if aboutzero(upper):
        return 2 * math.fabs(mass - lower)
    if aboutzero(lower):
        return 2 * math.fabs(mass - upper)
    return math.fabs(upper - lower)