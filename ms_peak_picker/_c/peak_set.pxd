
cdef class PeakBase(object):
    cdef:
        public double mz
        public double intensity
        public double area

    cpdef PeakBase clone(self)


cdef class FittedPeak(PeakBase):
    cdef:
        public double signal_to_noise
        public double full_width_at_half_max
        public double left_width
        public double right_width
        public long peak_count
        public long index

    cpdef bint _eq(self, FittedPeak other)

    @staticmethod
    cdef FittedPeak _create(double mz, double intensity, double signal_to_noise,
                            double full_width_at_half_max, double left_width,
                            double right_width, long peak_count, long index,
                            double area)

cdef class PeakSet(object):
    cdef:
        public tuple peaks

    cdef FittedPeak _get_nearest_peak(self, double mz, double* errout)

    cdef FittedPeak _has_peak(self, double mz, double tolerance=*)

    cdef PeakSet _between(self, double m1, double m2)

    cdef FittedPeak getitem(self, size_t i)

    cdef size_t get_size(self)
    
    cpdef bint _eq(self, PeakSet other)


cpdef FittedPeak binary_search_ppm_error(tuple array, double value, double tolerance)
cdef FittedPeak _binary_search_ppm_error(tuple array, double value, size_t lo, size_t hi, double tolerance)
cdef FittedPeak _binary_search_nearest_match(tuple array, double value, size_t lo, size_t hi, double* errout)


cdef class PeakSetIndexed(PeakSet):
    cdef:
        double* mz_index

    cpdef _allocate_index(self)

cdef size_t double_binary_search_ppm(double* array, double value, double tolerance, size_t n)