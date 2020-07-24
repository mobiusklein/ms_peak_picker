# cython: embedsignature=True

cimport cython
from cython cimport parallel
cimport numpy as np
from libc cimport math
from libc.math cimport fabs
import numpy as np

from ms_peak_picker._c.search cimport get_nearest
from ms_peak_picker._c.double_vector cimport (
    DoubleVector, make_double_vector_with_size,
    make_double_vector, double_vector_resize,
    double_vector_append, free_double_vector,
    print_double_vector, double_vector_to_list,
    list_to_double_vector, reset_double_vector)

from ms_peak_picker._c.peak_set cimport FittedPeak

from cpython.object cimport PyObject
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from cpython.sequence cimport PySequence_Fast, PySequence_Fast_ITEMS


from ms_peak_picker._c.peak_set cimport FittedPeak
from ms_peak_picker._c.search cimport get_nearest_binary

np.import_array()

cdef DTYPE_t minimum_signal_to_noise = 4.0


cdef object np_zeros = np.zeros


@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bint isclose(DTYPE_t x, DTYPE_t y, DTYPE_t rtol=1.e-5, DTYPE_t atol=1.e-8) nogil:
    return math.fabs(x-y) <= (atol + rtol * math.fabs(y))


@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bint aboutzero(DTYPE_t x) nogil:
    return isclose(x, 0)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t find_signal_to_noise(double target_val, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array, size_t index):
    cdef:
        DTYPE_t min_intensity_left, min_intensity_right
        size_t size, i
        DTYPE_t* pintensity_array
        int path_id

    with nogil:
        min_intensity_left = 0
        min_intensity_right = 0
        size = intensity_array.shape[0] - 1
        if aboutzero(target_val):
            return 0
        if index <= 0 or index >= size:
            return 0

        pintensity_array = &intensity_array[0]

        # locate the next valley to the left
        for i in range(index, 0, -1):
            if pintensity_array[i + 1] >= pintensity_array[i] and pintensity_array[i - 1] > pintensity_array[i]:
                min_intensity_left = pintensity_array[i]
                break
        else:
            min_intensity_left = pintensity_array[0]
        # locate the next valley to the right
        for i in range(index, size):
            if pintensity_array[i + 1] >= pintensity_array[i] and pintensity_array[i - 1] > pintensity_array[i]:
                min_intensity_right = pintensity_array[i]
                break
        else:
            min_intensity_right = pintensity_array[size]

        if aboutzero(min_intensity_left):
            if aboutzero(min_intensity_right):
                # The peak has no neighboring signal, so noise level is incalculable.
                # This may be really clean signal, or we may need to estimate noise over
                # a larger area.
                return target_val
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
    """
    Fit a least squares polynomial regression

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
    cdef:
        np.ndarray[DTYPE_t, ndim=1] weights
        np.ndarray[DTYPE_t, ndim=2] At, Z, At_T, At_At_T, I_At_At_T, At_Ai_At
        np.ndarray[DTYPE_t, ndim=2] B, out
        DTYPE_t mse, yfit, xpow
        size_t i, j

    weights = np.ones(n)

    # Weighted powers of x transposed
    # the polynomial regression's design matrix
    At = np_zeros((nterms + 1, n))
    for i in range(n):
        # set the intercept
        At[0, i] = weights[i]
        for j in range(1, nterms + 1):
            # the successive powers of x[i]
            At[j, i] = At[j - 1, i] * x[i]

    Z = np_zeros((n, 1))
    for i in range(n):
        Z[i, 0] = weights[i] * y[i]

    At_T = At.T
    At_At_T = At.dot(At_T)
    I_At_At_T = np.linalg.inv(At_At_T)
    At_Ai_At = I_At_At_T.dot(At)
    B = At_Ai_At.dot(Z)
    mse = 0
    out = np_zeros((2, n))
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

    # weights = np.ones(n)

    # Weighted powers of x transposed
    # Like Vandermonte Matrix?
    At = np_zeros((nterms + 1, n))
    for i in range(n):
        # At[0, i] = weights[i]
        At[0, i] = 1.0
        for j in range(1, nterms + 1):
            At[j, i] = At[j - 1, i] * x.v[i]

    Z = np_zeros((n, 1))
    for i in range(n):
        # Z[i, 0] = weights[i] * y.v[i]
        Z[i, 0] = 1.0 * y.v[i]

    At_T = At.T
    At_At_T = At.dot(At_T)
    try:
        I_At_At_T = np.linalg.inv(At_At_T)
    except Exception:
        for i in range(n):
            terms[i] = 0
        return -1
    At_Ai_At = I_At_At_T.dot(At)
    B = At_Ai_At.dot(Z)
    mse = 0
    # out = np_zeros((2, n))
    for i in range(n):
        terms[0] = B[0, 0]
        yfit = B[0, 0]
        xpow = x.v[i]
        for j in range(1, nterms):
            terms[j] = B[j, 0]
            yfit += B[j, 0] * xpow
            xpow = xpow * x.v[i]
        # out[0, i] = yfit
        # out[1, i] = y.v[i] - yfit
        mse += y.v[i] - yfit
    return mse


@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t find_right_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                               size_t data_index, DTYPE_t signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse, last_Y1
        DTYPE_t lower, current_mass
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1, mode='c'] coef
        DoubleVector *vect_mzs
        DoubleVector *vect_intensity
    points = 0
    peak = intensity_array[data_index]
    peak_half = peak / 2.
    mass = mz_array[data_index]

    vect_mzs = NULL
    vect_intensity = NULL

    coef = np_zeros(2)

    if peak == 0.0:
        return 0.

    size = (mz_array.shape[0]) - 1
    if data_index <= 0 or data_index >= size:
        return 0.

    last_Y1 = peak
    lower = mz_array[size]
    for index in range(data_index, size):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or (Y1 > last_Y1) or (
                (index > size - 1 or intensity_array[index + 1] > Y1) and (
                index > size - 2 or intensity_array[index + 2] > Y1) and signal_to_noise < minimum_signal_to_noise)):
            Y2 = intensity_array[index - 1]
            X1 = mz_array[index]
            X2 = mz_array[index - 1]

            if((Y2 - Y1 != 0) and (Y1 < peak_half)):
                # Linear interpolation approximation
                lower = X1 - (X1 - X2) * (peak_half - Y1) / (Y2 - Y1)
            else:
                lower = X1
                points = index - data_index + 1

                # Polynomial regression approximation
                if points >= 3:
                    if vect_mzs == NULL:
                        vect_mzs = make_double_vector()
                        vect_intensity = make_double_vector()

                    for k in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[index - k])
                        double_vector_append(vect_intensity, intensity_array[index - k])
                    j = 0
                    while (j < points) and (vect_intensity.v[0] == vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        if vect_mzs != NULL:
                            free_double_vector(vect_mzs)
                            free_double_vector(vect_intensity)
                        return 0.0

                    # coef will contain the result
                    mse = curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    reset_double_vector(vect_intensity)
                    reset_double_vector(vect_mzs)
                    lower = coef[1] * peak_half + coef[0]
            break
        last_Y1 = Y1
    if vect_mzs != NULL:
        free_double_vector(vect_mzs)
        free_double_vector(vect_intensity)
    return abs(lower - mass)



@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef DTYPE_t find_left_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                              size_t data_index, DTYPE_t signal_to_noise=0.):
    cdef:
        int points
        DTYPE_t peak, peak_half, mass, X1, X2, Y1, Y2, mse, last_Y1
        DTYPE_t upper, current_mass
        size_t size, index, j, k
        np.ndarray[DTYPE_t, ndim=1, mode='c'] coef
        DoubleVector *vect_mzs
        DoubleVector *vect_intensity

    points = 0
    peak = intensity_array[data_index]
    peak_half = peak / 2.
    mass = mz_array[data_index]

    coef = np_zeros(2)

    if peak == 0.0:
        return 0.

    vect_mzs = NULL
    vect_intensity = NULL

    size = (mz_array.shape[0]) - 1
    if data_index <= 0 or data_index >= size:
        return 0.
    last_Y1 = peak
    upper = mz_array[0]
    for index in range(data_index, -1, -1):
        current_mass = mz_array[index]
        Y1 = intensity_array[index]
        if ((Y1 < peak_half) or (abs(mass - current_mass) > 1.5) or (Y1 > last_Y1) or (
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
                    if vect_mzs == NULL:
                        vect_mzs = make_double_vector()
                        vect_intensity = make_double_vector()

                    for j in range(points - 1, -1, -1):
                        double_vector_append(vect_mzs, mz_array[data_index - j])
                        double_vector_append(vect_intensity, intensity_array[data_index - j])

                    j = 0
                    while j < points and isclose(vect_intensity.v[0], vect_intensity.v[j]):
                        j += 1

                    if j == points:
                        if vect_mzs != NULL:
                            free_double_vector(vect_mzs)
                            free_double_vector(vect_intensity)
                        return 0.

                    # coef will contain the results
                    mse = curve_reg_dv(vect_intensity, vect_mzs, points, coef, 1)
                    reset_double_vector(vect_intensity)
                    reset_double_vector(vect_mzs)
                    upper = coef[1] * peak_half + coef[0]
            break
        last_Y1 = Y1
    if vect_mzs != NULL:
        free_double_vector(vect_mzs)
        free_double_vector(vect_intensity)
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

    coef = np_zeros(2)

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
@cython.cdivision(True)
cpdef DTYPE_t quadratic_fit(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                            ssize_t index):
    cdef:
        DTYPE_t x1, x2, x3
        DTYPE_t y1, y2, y3
        DTYPE_t d, mz_fit

    if index < 1:
        return mz_array[0]
    elif index > mz_array.shape[0] - 1:
        return mz_array[-1]
    x1 = mz_array[index - 1]
    x2 = mz_array[index]
    x3 = mz_array[index + 1]
    y1 = intensity_array[index - 1]
    y2 = intensity_array[index]
    y3 = intensity_array[index + 1]

    d = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
    if d == 0:  # If the interpolated intensity is 0, the peak fitting is no better than the peak
        return x2
    mz_fit = ((x1 + x2) - ((y2 - y1) * (x3 - x2) * (x1 - x3)) / d) / 2.0
    return mz_fit


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double lorentzian_least_squares(DTYPE_t[:] mz_array,
                                     DTYPE_t[:] intensity_array,
                                     double amplitude, double full_width_at_half_max,
                                     double vo, size_t lstart, size_t lstop) nogil:

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
cpdef double lorentzian_fit(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array, np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                            size_t index, double full_width_at_half_max):
    cdef:
        double amplitude
        DTYPE_t vo, step, current_error, last_error
        size_t lstart, lstop, i
        DTYPE_t[:] view_mz_array
        DTYPE_t[:] view_intensity_array

    amplitude = intensity_array[index]
    vo = mz_array[index]
    step = math.fabs((vo - mz_array[index + 1]) / 100.0)

    if index < 1:
        return mz_array[index]
    elif index >= mz_array.shape[0] - 1:
        return mz_array[-1]

    lstart = get_nearest(mz_array, vo + full_width_at_half_max, index) + 1
    lstop = get_nearest(mz_array, vo - full_width_at_half_max, index) - 1

    view_mz_array = mz_array
    view_intensity_array = intensity_array

    with nogil:
        current_error = lorentzian_least_squares(
            view_mz_array, view_intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        for i in range(50):
            last_error = current_error
            vo = vo + step
            current_error = lorentzian_least_squares(
                view_mz_array, view_intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
            if (current_error > last_error):
                break

        vo = vo - step
        current_error = lorentzian_least_squares(
            view_mz_array, view_intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
        for i in range(50):
            last_error = current_error
            vo = vo - step
            current_error = lorentzian_least_squares(
                view_mz_array, view_intensity_array, amplitude, full_width_at_half_max, vo, lstart, lstop)
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


@cython.cdivision(True)
cpdef double gaussian_predict(FittedPeak peak, double mz) nogil:
    cdef:
        double x, center, amplitude, fwhm, spread, y
    x = mz
    center = peak.mz
    amplitude = peak.intensity
    fwhm = peak.full_width_at_half_max
    spread = fwhm / 2.35482
    y = amplitude * math.exp(-((x - center) ** 2) / (2 * spread ** 2))
    return y


cpdef object gaussian_shape(FittedPeak peak):
    cdef:
        double center, amplitude, fwhm, spread
        np.ndarray[double, ndim=1] x, y
    center = peak.mz
    amplitude = peak.intensity
    fwhm = peak.full_width_at_half_max
    spread = fwhm / 2.35482
    x = np.arange(center - fwhm - 0.02, center + fwhm + 0.02, 0.0001)
    y = amplitude * np.exp(-((x - center) ** 2) / (2 * spread ** 2))
    return x, y


cpdef double gaussian_error(FittedPeak peak, double mz, double intensity):
    y = gaussian_predict(peak, mz)
    return intensity - y


cpdef double gaussian_volume(FittedPeak peak):
    cdef:
        np.ndarray[double, ndim=1] x, y
    x, y = gaussian_shape(peak)
    return np.trapz(y, x, dx=0.0001)


cdef class PeakShapeModel(object):

    def __init__(self, peak):
        self.peak = peak
        self.center = peak.mz

    cpdef double predict(self, double mz):
        raise NotImplementedError()

    cpdef object shape(self):
        raise NotImplementedError()

    cpdef double volume(self):
        raise NotImplementedError()

    cpdef double error(self, double mz, double intensity):
        raise NotImplementedError()

    def __repr__(self):
        return "{self.__class__.__name__}({self.peak})".format(self=self)


cdef class GaussianModel(PeakShapeModel):
    cpdef object shape(self):
        return gaussian_shape(self.peak)

    cpdef double predict(self, double mz):
        return gaussian_predict(self.peak, mz)

    cpdef double volume(self):
        return gaussian_volume(self.peak)

    cpdef double error(self, double mz, double intensity):
        return gaussian_error(self.peak, mz, intensity)


cdef size_t find_starting_index(double* array, double value, double error_tolerance, size_t n) nogil:
    cdef:
        size_t lo, hi, mid, i, best_index
        double mid_value, error, best_error

    if n == 0:
        return 0
    lo = 0
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        mid_value = array[mid]
        error = (mid_value - value)
        if (fabs(error) < error_tolerance) or ((hi - 1) == lo):
            i = 0
            best_index = mid
            best_error = fabs(error)
            while mid - i  >= 0 and (mid - i) != (<size_t>-1):
                mid_value = array[mid - i]
                error = fabs(value - mid_value)
                if error > best_error:
                    break
                else:
                    best_index = mid - i
                    best_error = error
                i += 1
            return best_index
        elif error > 0:
            hi = mid
        else:
            lo = mid
    return 0

DEF PEAK_SHAPE_WIDTH = 1

cdef class PeakSetReprofiler(object):

    def __init__(self, models, dx=0.01):
        self.models = sorted(models, key=lambda x: x.center)
        self.dx = dx
        self.gridx = None
        self.gridy = None
        self._build_grid()

    cdef void _build_grid(self):
        if PyList_GET_SIZE(self.models) == 0:
            self.gridx = np.arange(0, 1, self.dx, dtype=np.float64)
            self.gridy = np.zeros_like(self.gridx, dtype=np.float64)
        else:
            lo = self.models[0].center
            hi = self.models[-1].center
            self.gridx = np.arange(max(lo - 3, 0), hi + 3, self.dx, dtype=np.float64)
            self.gridy = np.zeros_like(self.gridx, dtype=np.float64)

    @cython.boundscheck(False)
    cpdef _reprofile(PeakSetReprofiler self):
        cdef:
            size_t i, j, k, nmodels, n_pts
            double x, y, pred, model_center
            np.ndarray[double, ndim=1, mode='c'] gridx, gridy
            list models
            PyObject** pmodels
            PyObject* model
            double* xdata
            double* ydata

        gridx = self.gridx
        gridy = self.gridy
        models = self.models
        nmodels = PyList_GET_SIZE(models)
        if nmodels == 0:
            return
        xdata = &gridx[0]
        ydata = &gridy[0]

        n_pts = gridx.shape[0]
        pmodels = PySequence_Fast_ITEMS(PySequence_Fast(models, "error"))
        for i in range(nmodels):
            model = pmodels[i]
            model_center = (<PeakShapeModel>model).center
            k = j = find_starting_index(xdata, model_center, 1e-3, n_pts)
            while j < n_pts:
                x = gridx[j]
                pred = (<PeakShapeModel>model).predict(x)
                if pred == 0:
                    break
                ydata[j] += pred
                j += 1
            j = k - 1
            while j >= 0 and j != (<size_t>-1):
                x = gridx[j]
                pred = (<PeakShapeModel>model).predict(x)
                if pred == 0:
                    break
                ydata[j] += pred
                j -= 1


    cpdef size_t _find_starting_model(self, double x):
        cdef:
            size_t lo, hi, mid
            list models
            double center, err
            PeakShapeModel model
        lo = 0
        models = self.models
        hi = PyList_GET_SIZE(models)
        while (lo != hi):
            mid = (hi + lo) // 2
            model = <PeakShapeModel>PyList_GET_ITEM(models, mid)
            center = model.center
            err = center - x
            if abs(err) < 0.0001:
                return mid
            elif (hi - lo) == 1:
                return mid
            elif err > 0:
                hi = mid
            else:
                lo = mid
        return 0

    def reprofile(self):
        self._reprofile()
        return self.gridx, self.gridy


@cython.boundscheck(False)
cpdef zero_pad(DTYPE_t[:] x, DTYPE_t[:] y, DTYPE_t delta=0.05):
    cdef:
        size_t i, n
        DoubleVector* filled_x
        DoubleVector* filled_y
        DTYPE_t xi
        np.ndarray[double, ndim=1, mode='c'] outx, outy

    n = len(x)
    if len(y) != n:
        raise ValueError("Input arrays must have the same length")
    with nogil:
        filled_x = make_double_vector_with_size(n)
        filled_y = make_double_vector_with_size(n)
        for i in range(n):
            xi = x[i]
            if i == 0:
                double_vector_append(filled_x, xi - delta)
                double_vector_append(filled_y, 0.0)
            else:
                if (xi - x[i - 1]) > delta:
                    double_vector_append(filled_x, xi - delta)
                    double_vector_append(filled_y, 0.0)
            double_vector_append(filled_x, xi)
            double_vector_append(filled_y, y[i])
            if i == n - 1:
                double_vector_append(filled_x, xi + delta)
                double_vector_append(filled_y, 0.0)
            else:
                if (x[i + 1] - xi) > delta:
                    double_vector_append(filled_x, xi + delta)
                    double_vector_append(filled_y, 0.0)

    n = filled_x.used
    outx = np.empty(n, dtype=float)
    outy = np.empty(n, dtype=float)
    for i in range(n):
        outx[i] = filled_x.v[i]
        outy[i] = filled_y.v[i]
    free_double_vector(filled_x)
    free_double_vector(filled_y)
    return outx, outy

