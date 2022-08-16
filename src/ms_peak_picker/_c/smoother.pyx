# cython: embedsignature=True
#

from libc.math cimport sqrt, exp, pow

from ms_peak_picker._c.double_vector cimport (
    make_double_vector_with_size, make_double_vector,
    double_vector_append, reset_double_vector,
    free_double_vector, DoubleVector)

cimport cython
cimport numpy as np
import numpy as np

np.import_array()


cdef double pi = np.pi


@cython.cdivision(True)
cdef int mean_var(double* x, size_t n, double* mean, double* variance) nogil:
    cdef:
        size_t i
        double acc, m

    acc = 0
    for i in range(n):
        acc += x[i]
    m = mean[0] = acc / n

    acc = 0
    for i in range(n):
        acc += pow(x[i] - m, 2)
    variance[0] = acc / n
    return 0


@cython.cdivision(True)
cdef int vgauss(double* x, size_t n, double mean, double variance, double scale, DoubleVector* out) nogil:
    cdef:
        size_t i
        int err
        double val, a, b

    a = 1. / sqrt(2 * pi * variance)
    b = scale * variance

    for i in range(n):
        err = double_vector_append(out, a * exp((-pow(x[i] - mean, 2)) / b))
        if err != 0:
            return err



@cython.cdivision(True)
@cython.boundscheck(False)
cpdef np.ndarray[double, ndim=1, mode='c'] gaussian_smooth(
        np.ndarray[double, ndim=1, mode='c'] x,
        np.ndarray[double, ndim=1, mode='c'] y, double width=0.05):
    cdef:
        np.ndarray[double, ndim=1, mode='c'] smoothed
        size_t i, j, n, low_edge, high_edge, delta
        double center, spread, xj, thresh, acc, weights_acc
        double* x_part
        double* y_part
        DoubleVector* weights_vector

    smoothed = np.zeros_like(y)

    n = x.shape[0]
    weights_vector = make_double_vector()
    for i in range(n):
        thresh = x[i] - width
        j = i
        while j > 0 and x[j] > thresh:
            j -= 1
        low_edge = j
        j = i
        thresh = x[i] + width
        while j < n and x[j] < thresh:
            j += 1
        high_edge = j
        if high_edge - 1 == low_edge or high_edge == low_edge:
            smoothed[i] = y[i]
            continue
        delta = high_edge - low_edge
        x_part = &x[low_edge]
        y_part = &y[low_edge]
        mean_var(x_part, delta, &center, &spread)
        reset_double_vector(weights_vector)
        vgauss(x_part, delta, center, spread, 2.0, weights_vector)
        acc = 0
        weights_acc = 0
        for j in range(delta):
            acc += y_part[j] * weights_vector.v[j]
            weights_acc += weights_vector.v[j]
        smoothed[i] = acc / weights_acc
    free_double_vector(weights_vector)
    return smoothed
