cimport numpy as np
from ms_peak_picker._c.peak_set cimport PeakSet, FittedPeak


cdef class PeakIndex(object):
    cdef:
        public np.ndarray mz_array
        public np.ndarray intensity_array
        public PeakSet peaks

    cdef FittedPeak _has_peak(self, double mz, double tolerance=*)
    cdef PeakSet _between(self, double start, double stop)

    cpdef tuple all_peaks_for(self, double mz, double error_tolerance=*)

    cdef size_t get_size(self)

    cpdef PeakIndex clone(self)
    cpdef PeakIndex copy(self)