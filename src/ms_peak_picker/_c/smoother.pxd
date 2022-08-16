cimport cython
cimport numpy as np
from ms_peak_picker._c.double_vector cimport DoubleVector

cdef int vgauss(double* x, size_t n, double mean, double variance, double scale, DoubleVector* out) nogil
cdef int mean_var(double* x, size_t n, double* mean, double* variance) nogil

cpdef np.ndarray[double, ndim=1, mode='c'] gaussian_smooth(
        np.ndarray[double, ndim=1, mode='c'] x,
        np.ndarray[double, ndim=1, mode='c'] y, double width=*)
