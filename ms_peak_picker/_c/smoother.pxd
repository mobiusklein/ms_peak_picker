cimport cython
cimport numpy as np

cdef int vgauss(double* x, size_t n, double mean, double variance, double scale, DoubleVector* out) nogil
cdef int mean_var(double* x, size_t n, double* mean, double* variance) nogil

cpdef np.ndarray[double, ndim=1, mode='c'] gaussian_smooth(
        np.ndarray[double, ndim=1, mode='c'] x,
        np.ndarray[double, ndim=1, mode='c'] y, double width=*)
