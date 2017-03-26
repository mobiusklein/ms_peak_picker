# cython: embedsignature=True

cimport cython
cimport numpy as np
from libc cimport math
import numpy as np

from ms_peak_picker._c.search cimport get_nearest
from ms_peak_picker._c.double_vector cimport (
    DoubleVector, make_double_vector_with_size,
    make_double_vector, double_vector_resize,
    double_vector_append, free_double_vector,
    print_double_vector, double_vector_to_list,
    list_to_double_vector)


cdef DTYPE_t minimum_signal_to_noise = 4.0


@cython.nonecheck(False)
@cython.cdivision(True)
cdef bint isclose(DTYPE_t x, DTYPE_t y, DTYPE_t rtol=1.e-5, DTYPE_t atol=1.e-8):
    return abs(x-y) <= (atol + rtol * abs(y))


@cython.nonecheck(False)
@cython.cdivision(True)
cdef bint aboutzero(DTYPE_t x):
    return isclose(x, 0)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t find_signal_to_noise(double target_val, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array, size_t index):
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


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t curve_reg(np.ndarray[DTYPE_t, ndim=1, mode='c'] x, np.ndarray[DTYPE_t, ndim=1, mode='c'] y, size_t n,
                        np.ndarray[DTYPE_t, ndim=1, mode='c'] terms, size_t nterms):
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
        mse += y[i] - yfit
    return mse



@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef DTYPE_t curve_reg_dv(DoubleVector* x, DoubleVector* y, size_t n,
                        np.ndarray[DTYPE_t, ndim=1, mode='c'] terms, size_t nterms):
    cdef:
        np.ndarray[DTYPE_t, ndim=1] weights
        np.ndarray[DTYPE_t, ndim=2] At, Z, At_Ai_At
        np.ndarray[DTYPE_t, ndim=2] At_T, At_At_T, I_At_At_T
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
            At[j, i] = At[j - 1, i] * x.v[i]
    
    Z = np.zeros((n, 1))
    for i in range(n):
        Z[i, 0] = weights[i] * y.v[i]
    
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
        xpow = x.v[i]
        for j in range(1, nterms):
            terms[j] = B[j, 0]
            yfit += B[j, 0] * xpow
            xpow = xpow * x.v[i]
        out[0, i] = yfit
        out[1, i] = y.v[i] - yfit
        mse += y.v[i]-yfit
    return mse


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t find_right_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                               size_t data_index, DTYPE_t signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse
        DTYPE_t lower, current_mass
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1, mode='c'] coef
        DoubleVector *vect_mzs
        DoubleVector *vect_intensity
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

    lower = mz_array[size]
    for index in range(data_index, size):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or (
                (index > size - 1 or intensity_array[index + 1] > Y1) and (
                index > size - 2 or intensity_array[index + 2] > Y1) and signal_to_noise < minimum_signal_to_noise)):
            Y2 = intensity_array[index - 1]
            X1 = mz_array[index]
            X2 = mz_array[index - 1]

            if((Y2 - Y1 != 0) and (Y1 < peak_half)):
                lower = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                lower = X1
                points = index - data_index + 1

                if points >= 3:
                    vect_mzs = make_double_vector()
                    vect_intensity = make_double_vector()

                    for k in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[index - k])
                        double_vector_append(vect_intensity, intensity_array[index - k])
                    j = 0
                    while (j < points) and (vect_intensity.v[0] == vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        return 0.0

                    # coef will contain the result
                    curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    free_double_vector(vect_intensity)
                    free_double_vector(vect_mzs)
                    lower = coef[1] * peak_half + coef[0]
            break
    return abs(lower - mass)



@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t find_left_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                              size_t data_index, DTYPE_t signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse
        DTYPE_t upper, current_mass
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1, mode='c'] coef
        DoubleVector *vect_mzs
        DoubleVector *vect_intensity

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
        if ((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or (
                (index < 1 or intensity_array[index - 1] > Y1) and (
                    index < 2 or intensity_array[index - 2] > Y1) and (signal_to_noise < minimum_signal_to_noise))):
            Y2 = intensity_array[index + 1]
            X1 = mz_array[index]
            X2 = mz_array[index + 1]

            if ((Y2 - Y1 != 0) and (Y1 < peak_half)):
                upper = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                upper = X1
                points = data_index - index + 1
                if points >= 3:
                    vect_mzs = make_double_vector()
                    vect_intensity = make_double_vector()

                    for j in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[index - j])
                        double_vector_append(vect_intensity, intensity_array[index - j])

                    j = 0
                    while j < points and (vect_intensity.v[0] == vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        return 0.

                    # coef will contain the results
                    curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    free_double_vector(vect_intensity)
                    free_double_vector(vect_mzs)
                    upper = coef[1] * peak_half + coef[0]
            break
    return abs(mass - upper)



@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t find_full_width_at_half_max(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                                          size_t data_index, double signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse
        DTYPE_t upper, lower, current_mass
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1, mode='c'] coef
        DoubleVector* vect_mzs
        DoubleVector* vect_intensity

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
        if ((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or (
                (index < 1 or intensity_array[index - 1] > Y1) and (
                 index < 2 or intensity_array[index - 2] > Y1) and (signal_to_noise < minimum_signal_to_noise))):
            Y2 = intensity_array[index + 1]
            X1 = mz_array[index]
            X2 = mz_array[index + 1]

            if (not aboutzero(Y2 - Y1) and (Y1 < peak_half)):
                upper = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                upper = X1
                points = data_index - index + 1
                if points >= 3:
                    
                    vect_mzs = make_double_vector()
                    vect_intensity = make_double_vector()

                    for j in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[data_index - j])
                        double_vector_append(vect_intensity, intensity_array[data_index - j])
                    
                    j = 0
                    while j < points and (vect_intensity.v[0] == vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        return 0.
                    mse = curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    upper = coef[1] * peak_half + coef[0]
                    free_double_vector(vect_mzs)
                    free_double_vector(vect_intensity)
            break
    lower = mz_array[size]
    for index in range(data_index, size):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or ((index > size - 1 or intensity_array[index + 1] > Y1) and (
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
                    vect_mzs = make_double_vector()
                    vect_intensity = make_double_vector()
                    
                    for k in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[index - k])
                        double_vector_append(vect_intensity, intensity_array[index - k])
                    j = 0
                    while (j < points) and (vect_intensity.v[0] == vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        return 0.0
                    mse = curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    free_double_vector(vect_intensity)
                    free_double_vector(vect_mzs)
                    lower = coef[1] * peak_half + coef[0]
            break

    if aboutzero(upper):
        return 2 * abs(mass - lower)
    if aboutzero(lower):
        return 2 * abs(mass - upper)
    return abs(upper - lower)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double lorenztian_least_squares(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array, double amplitude, double full_width_at_half_max,
                                     double vo, size_t lstart, size_t lstop):

    cdef:
        double root_mean_squared_error, u, Y2
        long Y1
        size_t index


    root_mean_squared_error = 0

    for index in range(lstart, lstop + 1):
        u = 2 / float(full_width_at_half_max) * (mz_array[index] - vo)
        Y1 = int(amplitude / float(1 + u * u))
        Y2 = intensity_array[index]

        root_mean_squared_error += (Y1 - Y2) * (Y1 - Y2)

    return root_mean_squared_error


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double lorenztian_fit(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                            size_t index, double full_width_at_half_max):
    cdef:
        double amplitude
        DTYPE_t vo, step, current_error, last_error
        size_t lstart, lstop, i

    amplitude = intensity_array[index]
    vo = mz_array[index]
    step = math.fabs((vo - mz_array[index + 1]) / 100.0)

    if index < 1:
        return mz_array[index]
    elif index >= mz_array.shape[0] - 1:
        return mz_array[-1]

    lstart = get_nearest(mz_array, vo + full_width_at_half_max, index) + 1
    lstop = get_nearest(mz_array, vo - full_width_at_half_max, index) - 1

    current_error = lorenztian_least_squares(
        mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
    for i in range(50):
        last_error = current_error
        vo = vo + step
        current_error = lorenztian_least_squares(
            mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        if (current_error > last_error):
            break

    vo = vo - step
    current_error = lorenztian_least_squares(
        mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
    for i in range(50):
        last_error = current_error
        vo = vo - step
        current_error = lorenztian_least_squares(
            mz_array, intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        if (current_error > last_error):
            break

    vo += step
    return vo


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double peak_area(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c']  intensity_array,
                       size_t start, size_t stop):
    cdef:
        double area
        size_t i
        DTYPE_t x1, y1, x2, y2

    area = 0.
    for i in range(start + 1, stop):
        x1 = mz_array[i - 1]
        y1 = intensity_array[i - 1]
        x2 = mz_array[i]
        y2 = intensity_array[i]
        area += (y1 * (x2 - x1)) + ((y2 - y1) * (x2 - x1) / 2.)

    return area
