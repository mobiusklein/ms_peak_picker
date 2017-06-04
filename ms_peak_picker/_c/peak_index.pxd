cimport numpy as np
from ms_peak_picker._c.peak_set cimport PeakSet, FittedPeak


cdef class PeakIndex(object):
    cdef:
        public np.ndarray mz_array
        public np.ndarray intensity_array
        public PeakSet peaks

    cdef FittedPeak _has_peak(self, double mz, double tolerance=*)
    cdef PeakSet _between(self, double start, double stop)
    cdef size_t get_size(self)