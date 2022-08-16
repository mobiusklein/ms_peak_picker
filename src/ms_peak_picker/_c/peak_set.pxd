
cdef class PeakBase(object):
    cdef:
        public double mz
        public double intensity
        public double area

    cpdef PeakBase clone(self)
    cpdef PeakBase copy(self)


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
        public bint indexed

    @staticmethod
    cdef PeakSet _create(tuple peaks)

    cpdef reindex(self)
    cpdef size_t _index(self)

    cdef FittedPeak _get_nearest_peak(self, double mz, double* errout)

    cdef FittedPeak _has_peak(self, double mz, double tolerance=*)

    cpdef PeakSet clone(self)
    cpdef PeakSet copy(self)

    cdef PeakSet _between(self, double m1, double m2)
    cdef int _between_bounds(self, double m1, double m2, size_t* startp, size_t* endp)

    cpdef tuple all_peaks_for(self, double mz, double error_tolerance=*)

    cdef FittedPeak getitem(self, size_t i)

    cdef size_t get_size(self)

    cpdef bint _eq(self, PeakSet other)


cpdef FittedPeak binary_search_ppm_error(tuple array, double value, double tolerance)
cdef FittedPeak _binary_search_ppm_error(tuple array, double value, size_t lo, size_t hi, double tolerance)
cdef FittedPeak _binary_search_nearest_match(tuple array, double value, size_t lo, size_t hi, double* errout)


cdef struct index_cell:
    double center_value
    size_t start
    size_t end


cdef struct index_list:
    index_cell* index
    size_t size
    double low
    double high


cdef int check_index(index_list* index) nogil


cdef size_t INTERVAL_INDEX_SIZE


cdef class PeakSetIndexed(PeakSet):
    cdef:
        double* mz_index
        index_list* interval_index

    cpdef _allocate_index(self, size_t interval_index_size)

    cpdef FittedPeak has_peak_hinted(self, double mz, double tolerance=*, size_t hint_start=*, size_t hint_end=*)

cdef size_t double_binary_search_ppm(double* array, double value, double tolerance, size_t n)
cdef size_t double_binary_search_nearest_match(double* array, double value, size_t n)
cdef size_t double_binary_search_ppm_with_hint(double* array, double value, double tolerance, size_t n, size_t hint)
