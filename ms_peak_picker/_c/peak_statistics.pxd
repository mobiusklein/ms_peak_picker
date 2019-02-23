cimport cython
cimport numpy as np

from ms_peak_picker._c.peak_set cimport FittedPeak

ctypedef np.float64_t DTYPE_t


cpdef DTYPE_t find_signal_to_noise(
    double target_val,
    np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
    size_t index)

cpdef DTYPE_t find_full_width_at_half_max(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                                          np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                                          size_t data_index,
                                          double signal_to_noise=*)

cpdef DTYPE_t find_right_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                               size_t data_index, DTYPE_t signal_to_noise=*)

cpdef DTYPE_t find_left_width(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                              np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                              size_t data_index, DTYPE_t signal_to_noise=*)

cpdef DTYPE_t quadratic_fit(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                            ssize_t index)

cpdef double lorentzian_fit(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                            size_t index, double full_width_at_half_max)

cpdef double peak_area(np.ndarray[DTYPE_t, ndim=1, mode='c'] mz_array,
                       np.ndarray[DTYPE_t, ndim=1, mode='c'] intensity_array,
                       size_t start, size_t stop)


cdef class PeakShapeModel(object):
    cdef:
        public FittedPeak peak
        public double center

    cpdef double predict(self, double mz)

    cpdef object shape(self)

    cpdef double volume(self)

    cpdef double error(self, double mz, double intensity)


cdef class GaussianModel(PeakShapeModel):
    pass


cdef class PeakSetReprofiler(object):
    cdef:
        public list models
        public np.ndarray gridx
        public np.ndarray gridy
        public double dx

    cpdef size_t _find_starting_model(self, double x)

    cdef void _build_grid(self)

    cpdef _reprofile(self)
