cimport cython
cimport numpy as np


ctypedef np.float64_t DTYPE_t

cpdef DTYPE_t find_signal_to_noise(double target_val, np.ndarray[DTYPE_t, ndim=1] intensity_array, size_t index)

cpdef DTYPE_t curve_reg(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, size_t n, np.ndarray[DTYPE_t, ndim=1] terms, size_t nterms)

cpdef DTYPE_t find_full_width_at_half_max(np.ndarray[DTYPE_t, ndim=1] mz_array, np.ndarray[DTYPE_t, ndim=1] intensity_array, size_t data_index,
                                          double signal_to_noise=*)

cdef double lorenztian_least_squares(np.ndarray[DTYPE_t, ndim=1] mz_array, np.ndarray[DTYPE_t, ndim=1] intensity_array, double amplitude, double full_width_at_half_max,
                                     double vo, size_t lstart, size_t lstop)

cpdef double lorenztian_fit(np.ndarray[DTYPE_t, ndim=1] mz_array, np.ndarray[DTYPE_t, ndim=1] intensity_array, size_t index, double full_width_at_half_max)

cpdef double peak_area(np.ndarray[DTYPE_t, ndim=1] mz_array, np.ndarray[DTYPE_t, ndim=1]  intensity_array, size_t start, size_t stop)
