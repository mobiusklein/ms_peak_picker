cimport cython
cimport numpy as np

cdef class PartialPeakFitState(object):
    cdef:
        public bint set
        public double left_width
        public double right_width
        public double full_width_at_half_max
        public double signal_to_noise

    cpdef reset(self)


cdef class PeakProcessor(object):
    cdef:
        public double background_intensity
        public double _intensity_threshold
        public double _signal_to_noise_threshold

        public PartialPeakFitState partial_fit_state

        public str fit_type
        public str peak_mode

        public list peak_data

        public bint threshold_data
        public bint verbose

    cpdef double get_signal_to_noise_threshold(self)
    cpdef object set_signal_to_noise_threshold(self, double signal_to_noise_threshold)

    cpdef double get_intensity_threshold(self)
    cpdef object set_intensity_threshold(self, double intensity_threshold)


    cpdef size_t _discover_peaks(self, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array, double start_mz, double stop_mz)
    cpdef double find_full_width_at_half_max(self, Py_ssize_t index, np.ndarray[cython.floating, ndim=1] mz_array,
                                             np.ndarray[cython.floating, ndim=1] intensity_array, double signal_to_noise)
    cpdef double fit_peak(self, Py_ssize_t index, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array)
    cpdef double area(self, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array, double mz,
                      double full_width_at_half_max, Py_ssize_t index)