# cython: embedsignature=True
cimport cython
import operator
from libc.stdlib cimport malloc, realloc, free

from libc.stdio cimport printf

from cpython.tuple cimport PyTuple_GET_ITEM, PyTuple_GetItem, PyTuple_GetSlice, PyTuple_GET_SIZE
from cpython cimport PyObject

cdef double ppm_error(double x, double y):
    return (x - y) / y


cdef double INF = float('inf')


@cython.freelist(1000000)
cdef class PeakBase:

    cpdef PeakBase clone(self):
        return PeakBase()

    cpdef PeakBase copy(self):
        return self.clone()


cdef class FittedPeak(PeakBase):
    """Represent a single centroided mass spectral peak.

    FittedPeak instances are comparable for equality and
    hashable.

    Attributes
    ----------
    mz : float
        The m/z value at which the peak achieves its maximal
        abundance
    intensity : float
        The apex height of the peak
    signal_to_noise : float
        The signal to noise ratio of the peak
    full_width_at_half_max : double
        The symmetric full width at half of the
        peak's maximum height
    index : int
        The index at which the peak was found in the m/z array
        it was picked from
    peak_count : int
        The order in which the peak was picked
    area : float
        The total area of the peak, as determined
        by trapezoid integration
    left_width : float
        The left-sided width at half of max
    right_width : float
        The right-sided width at half of max
    """
    def __init__(self, mz, intensity, signal_to_noise, peak_count, index, full_width_at_half_max,
                 area, left_width=0, right_width=0):
            self.mz = mz
            self.intensity = intensity
            self.signal_to_noise = signal_to_noise
            self.peak_count = peak_count
            self.index = index
            self.full_width_at_half_max = full_width_at_half_max
            self.area = area
            self.left_width = left_width
            self.right_width = right_width

    @staticmethod
    cdef FittedPeak _create(double mz, double intensity, double signal_to_noise,
                            double full_width_at_half_max, double left_width,
                            double right_width, long peak_count, long index,
                            double area):
        cdef:
            FittedPeak inst

        inst = FittedPeak.__new__(FittedPeak)
        inst.mz = mz
        inst.intensity = intensity
        inst.signal_to_noise = signal_to_noise
        inst.full_width_at_half_max = full_width_at_half_max
        inst.left_width = left_width
        inst.right_width = right_width
        inst.peak_count = peak_count
        inst.index = index
        inst.area = area
        return inst

    cpdef PeakBase clone(self):
        return FittedPeak._create(
            self.mz, self.intensity, self.signal_to_noise,
            self.full_width_at_half_max, self.left_width, self.right_width,
            self.peak_count, self.index, self.area)

    def __repr__(self):
        return ("FittedPeak(mz=%0.3f, intensity=%0.3f, signal_to_noise=%0.3f, peak_count=%d, "
                "index=%d, full_width_at_half_max=%0.3f, area=%0.3f)") % (
                self.mz, self.intensity, self.signal_to_noise,
                self.peak_count, self.index, self.full_width_at_half_max,
                self.area)

    def __getstate__(self):
        return (self.mz, self.intensity, self.signal_to_noise,
                self.peak_count, self.index, self.full_width_at_half_max,
                self.area, self.left_width, self.right_width)

    def __setstate__(self, state):
        (self.mz, self.intensity, self.signal_to_noise,
         self.peak_count, self.index, self.full_width_at_half_max,
         self.area, self.left_width, self.right_width) = state

    def __reduce__(self):
        return FittedPeak, (self.mz, self.intensity, self.signal_to_noise, self.peak_count,
                            self.index, self.full_width_at_half_max, self.left_width,
                            self.right_width)

    def __hash__(self):
        return hash(self.mz)

    cpdef bint _eq(self, FittedPeak other):
        return (abs(self.mz - other.mz) < 1e-5) and (
            abs(self.intensity - other.intensity) < 1e-5) and (
            abs(self.signal_to_noise - other.signal_to_noise) < 1e-5) and (
            abs(self.full_width_at_half_max - other.full_width_at_half_max) < 1e-5)

    def __richcmp__(self, other, int code):
        if code == 2:
            if other is None:
                return False
            return self._eq(other)
        elif code == 3:
            if other is None:
                return True
            return not self._eq(other)



cdef class PeakSet(object):
    """A sequence of :class:`FittedPeak` instances, ordered by m/z,
    providing efficient search and retrieval of individual peaks or
    whole intervals of the m/z domain.

    This collection is not meant to be updated once created, as it
    it is indexed for ease of connecting individual peak objects to
    their position in the underlying sequence.

    One :class:`PeakSet` is considered equal to another if all of their
    contained :class:`FittedPeak` members are equal to each other

    Attributes
    ----------
    peaks : tuple
        The :class:`FittedPeak` instances, stored
    """

    @staticmethod
    cdef PeakSet _create(tuple peaks):
        cdef:
            PeakSet inst
        inst = PeakSet.__new__(PeakSet)
        inst.peaks = peaks
        return inst

    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return PyTuple_GET_SIZE(self.peaks)

    cdef size_t get_size(self):
        return PyTuple_GET_SIZE(self.peaks)

    cpdef reindex(self):
        """Re-indexes the sequence of peaks, updating their
        :attr:`peak_count` and setting their :attr:`index` if
        it is missing.
        """
        self._index()

    cpdef size_t _index(self):
        cdef:
            size_t i, n
            FittedPeak peak
        self.peaks = tuple(sorted(self.peaks, key=operator.attrgetter('mz')))
        i = 0
        n = self.get_size()
        for i in range(n):
            peak = self.getitem(i)
            peak.peak_count = i
            if peak.index == -1:
                peak.index = i
        self.indexed = True
        return i

    def has_peak(self, double mz, double tolerance=1e-5):
        """Search the for the peak nearest to `mz` within
        `tolerance` error (in PPM)

        Parameters
        ----------
        mz : float
            The m/z to search for
        tolerance : float, optional
            The error tolerance to accept. Defaults to 1e-5 (10 ppm)

        Returns
        -------
        FittedPeak
            The peak, if found. `None` otherwise.
        """
        peak = self._has_peak(mz, tolerance)
        if peak is _null_peak:
            return None
        return peak

    def get_nearest_peak(self, double mz):
        cdef:
            FittedPeak peak
            double err
        peak = self._get_nearest_peak(mz, &err)
        return peak, err

    cdef FittedPeak _get_nearest_peak(self, double mz, double* errout):
        cdef:
            FittedPeak peak
        peak = _binary_search_nearest_match(self.peaks, mz, 0, PyTuple_GET_SIZE(self.peaks), errout)
        return peak

    cdef FittedPeak _has_peak(self, double mz, double tolerance=1e-5):
        cdef:
            FittedPeak peak
        peak = binary_search_ppm_error(self.peaks, mz, tolerance)
        return peak

    cdef FittedPeak getitem(self, size_t i):
        return <FittedPeak>PyTuple_GET_ITEM(self.peaks, i)

    def __repr__(self):
        return "<PeakSet %d Peaks>" % (len(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PeakSet(self.peaks[item])
        return self.peaks[item]

    cpdef PeakSet clone(self):
        """Creates a deep copy of the sequence of peaks

        Returns
        -------
        PeakSet
        """
        cdef PeakSet inst = PeakSet([p.clone() for p in self])
        inst.indexed = self.indexed
        return inst

    cpdef PeakSet copy(self):
        return self.clone()

    def between(self, double m1, double m2):
        """Retrieve a :class:`PeakSet` containing all the peaks
        in `self` whose m/z is between `m1` and `m2`.

        These peaks are not copied.

        Parameters
        ----------
        m1 : float
            The lower m/z limit
        m2 : float
            The upper m/z limit

        Returns
        -------
        PeakSet
        """
        return self._between(m1, m2)

    cdef PeakSet _between(self, double m1, double m2):
        cdef:
            FittedPeak p1
            FittedPeak p2
            double err
            tuple sliced
            size_t start, end, n
        p1 = self._get_nearest_peak(m1, &err)
        p2 = self._get_nearest_peak(m2, &err)
        start = p1.peak_count
        end = p2.peak_count + 1
        n = self.get_size()
        if p1.mz < m1 and start + 1 < n:
            start += 1
        if p2.mz > m2 and end > 0:
            end -= 1
        sliced = <tuple>PyTuple_GetSlice(self.peaks, start, end)
        return PeakSet._create(sliced)

    cdef int _between_bounds(self, double m1, double m2, size_t* startp, size_t* endp):
        cdef:
            FittedPeak p1
            FittedPeak p2
            double err
            size_t start, end, n

        p1 = self._get_nearest_peak(m1, &err)
        p2 = self._get_nearest_peak(m2, &err)
        start = p1.peak_count
        end = p2.peak_count + 1
        n = self.get_size()
        if p1.mz < m1 and start + 1 < n:
            start += 1
        if p2.mz > m2 and end > 0:
            end -= 1
        startp[0] = start
        endp[0] = end
        return 0

    cpdef tuple all_peaks_for(self, double mz, double error_tolerance=2e-5):
        """Find all peaks within `error_tolerance` ppm of `mz` m/z.

        Parameters
        ----------
        mz : float
            The query m/z
        error_tolerance : float, optional
            The parts-per-million error tolerance (the default is 2e-5)

        Returns
        -------
        tuple
        """
        cdef:
            double lo, hi
            size_t start, end
            tuple sliced

        lo = mz - (mz * error_tolerance)
        hi = mz + (mz * error_tolerance)
        self._between_bounds(lo, hi, &start, &end)
        sliced = <tuple>PyTuple_GetSlice(self.peaks, start, end)
        return sliced

    def __getstate__(self):
        return self.peaks

    def __setstate__(self, state):
        self.peaks = state
        self.indexed = False
        self.reindex()

    def __reduce__(self):
        return PeakSet, (tuple(),), self.__getstate__()

    cpdef bint _eq(self, PeakSet other):
        cdef:
            size_t i, n
            FittedPeak p1, p2

        n = self.get_size()
        if n != other.get_size():
            return False
        for i in range(n):
            p1 = self.getitem(i)
            p2 = other.getitem(i)
            if not p1._eq(p2):
                return False
        return True

    def __richcmp__(self, object other, int code):
        cdef:
            tuple other_tuple

        if other is None:
            if code == 2:
                return False
            elif code == 3:
                return True
        if isinstance(other, PeakSet):
            if code == 2:
                return self._eq(<PeakSet>other)
            elif code == 3:
                return not self._eq(<PeakSet>other)
            else:
                return NotImplemented
        else:
            try:
                other_tuple = tuple(other)
                if code == 2:
                    return self.peaks == other_tuple
                elif code == 3:
                    return self.peaks != other_tuple
                else:
                    return NotImplemented
            except Exception:
                if code == 2:
                    return False
                elif code == 3:
                    return True
                else:
                    return NotImplemented


cdef FittedPeak _null_peak
_null_peak = FittedPeak(-1,-1,-1,-1,-1,-1, -1)


cpdef FittedPeak binary_search_ppm_error(tuple array, double value, double tolerance):
    return _binary_search_ppm_error(array, value, 0, len(array), tolerance)


cdef FittedPeak _binary_search_ppm_error(tuple array, double value, size_t lo, size_t hi, double tolerance):
    cdef:
        size_t mid, lower_edge
        FittedPeak target
        double target_value, error
    if (hi - lo) < 5:
        return _sweep_solution_ppm_error(array, value, lo, hi, tolerance)
    else:
        mid = (hi + lo) // 2
        target = <FittedPeak>PyTuple_GetItem(array, mid)
        target_value = target.mz
        error = ppm_error(value, target_value)
        if abs(error) <= tolerance:
            return _sweep_solution_ppm_error(array, value, max(mid - (mid if mid < 5 else 5), lo), min(mid + 5, hi), tolerance)
        elif target_value > value:
            return _binary_search_ppm_error(array, value, lo, mid, tolerance)
        elif target_value < value:
            return _binary_search_ppm_error(array, value, mid, hi, tolerance)
    return _null_peak


cdef FittedPeak _sweep_solution_ppm_error(tuple array, double value, size_t lo, size_t hi, double tolerance):
    cdef:
        double best_error, error, abs_error
        size_t i, limit

    best_index = -1
    best_intensity = 0
    best_error = 1000000000000000
    for i in range(hi - lo):
        target = <FittedPeak>PyTuple_GetItem(array, lo + i)
        error = ppm_error(value, target.mz)
        abs_error = abs(error)
        if abs_error < tolerance and (abs_error < (best_error * 1.1)) and (target.intensity > best_intensity):
            best_index = lo + i
            best_error = abs_error
    if best_index == -1:
        return _null_peak
    else:
        return <FittedPeak>PyTuple_GetItem(array, best_index)


cdef FittedPeak _sweep_nearest_match(tuple array, double value, size_t lo, size_t hi, double* errout):
    cdef:
        size_t i
        size_t best_index
        double best_error, err
        double v

    best_error = float('inf')
    best_index = -1
    for i in range(hi - lo):
        i += lo
        v = array[i].mz
        err = abs(v - value)
        if err < best_error:
            best_error = err
            best_index = i
    errout[0] = best_error
    return array[best_index]


cdef FittedPeak _binary_search_nearest_match(tuple array, double value, size_t lo, size_t hi, double* errout):
    cdef:
        size_t mid
        double v
        double err

    if (hi - lo) < 5:
        return _sweep_nearest_match(array, value, lo, hi, errout)
    else:
        mid = (hi + lo) // 2
        v = array[mid].mz
        if abs(v - value) < 1.:
            return _sweep_nearest_match(array, value, lo, hi, errout)
        elif v > value:
            return _binary_search_nearest_match(array, value, lo, mid, errout)
        else:
            return _binary_search_nearest_match(array, value, mid, hi, errout)


@cython.cdivision(True)
cdef size_t double_binary_search_ppm(double* array, double value, double tolerance, size_t n):
    cdef:
        size_t lo, hi, mid
        size_t i, best_ix
        double x, err, best_err, abs_err
    lo = 0
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        x = array[mid]
        err = (x - value) / value
        abs_err = abs(err)
        if abs_err < tolerance:
            i = mid
            best_error = abs_err
            best_ix = mid
            while i > 0:
                i -= 1
                x = array[i]
                err = (x - value) / value
                abs_err = abs(err)
                if abs_err > tolerance:
                    break
                elif abs_err < best_error:
                    best_error = abs_err
                    best_ix = i
            i = mid
            while i < n - 1:
                i += 1
                x = array[i]
                err = (x - value) / value
                abs_err = abs(err)
                if abs_err > tolerance:
                    break
                elif abs_err < best_error:
                    best_error = abs_err
                    best_ix = i
            return best_ix
        elif (hi - 1) == lo:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


@cython.cdivision(True)
cdef size_t double_binary_search_nearest_match(double* array, double value, size_t n):
    cdef:
        size_t lo, hi, mid
        size_t i, best_ix
        double x, err, best_err, abs_err
    lo = 0
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        x = array[mid]
        err = x - value
        if err == 0 or ((hi - 1) == lo):
            i = mid
            best_err = abs(err)
            best_ix = mid
            while i > 0:
                i -= 1
                x = array[i]
                err = (x - value)
                abs_err = abs(err)
                if abs_err > best_err:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            i = mid
            while i < n - 1:
                i += 1
                x = array[i]
                err = (x - value)
                abs_err = abs(err)
                if abs_err > best_err:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            return best_ix
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


@cython.cdivision(True)
cdef size_t double_binary_search_ppm_with_hint(double* array, double value, double tolerance, size_t n, size_t hint):
    cdef:
        size_t lo, hi, mid
        size_t i, best_ix
        double x, err, best_err, abs_err
    lo = hint
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        x = array[mid]
        err = (x - value) / value
        abs_err = abs(err)
        if abs_err < tolerance:
            i = mid
            best_err = abs_err
            best_ix = mid
            while i > 0:
                i -= 1
                x = array[i]
                err = (x - value) / value
                abs_err = abs(err)
                if abs_err > tolerance:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            i = mid
            while i < n - 1:
                i += 1
                x = array[i]
                err = (x - value) / value
                abs_err = abs(err)
                if abs_err > tolerance:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            return best_ix
        elif (hi - 1) == lo:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0

@cython.cdivision(True)
cdef size_t double_binary_search_nearest_match_with_hint(double* array, double value, size_t n, size_t hint):
    cdef:
        size_t lo, hi, mid
        size_t i, best_ix
        double x, err, best_err, abs_err
    lo = hint
    hi = n
    while hi != lo:
        mid = (hi + lo) // 2
        x = array[mid]
        err = x - value
        if err == 0 or ((hi - 1) == lo):
            i = mid
            best_err = abs(err)
            best_ix = mid
            while i > 0:
                i -= 1
                x = array[i]
                err = (x - value)
                abs_err = abs(err)
                if abs_err > best_err:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            i = mid
            while i < n - 1:
                i += 1
                x = array[i]
                err = (x - value)
                abs_err = abs(err)
                if abs_err > best_err:
                    break
                elif abs_err < best_err:
                    best_err = abs_err
                    best_ix = i
            return best_ix
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


cdef size_t INTERVAL_INDEX_SIZE = 0

cdef class PeakSetIndexed(PeakSet):

    @staticmethod
    cdef PeakSet _create(tuple peaks):
        cdef:
            PeakSetIndexed inst
        inst = PeakSetIndexed.__new__(PeakSetIndexed)
        inst.peaks = peaks
        inst.mz_index = NULL
        inst.interval_index = NULL
        return inst

    def __init__(self, peaks):
        PeakSet.__init__(self, peaks)
        self.mz_index = NULL
        self.interval_index = NULL

    def __dealloc__(self):
        if self.mz_index != NULL:
            free(self.mz_index)
        if self.interval_index != NULL:
            free_index_list(self.interval_index)

    cpdef PeakSet clone(self):
        cdef PeakSetIndexed inst = PeakSetIndexed(tuple([p.clone() for p in self]))
        if self.indexed:
            inst.reindex()
        return inst

    cpdef _allocate_index(self, size_t interval_index_size):
        cdef:
            size_t i, n
            FittedPeak p
            index_list* interval_index
        n = self.get_size()
        if self.mz_index != NULL:
            free(self.mz_index)
            self.mz_index = NULL
        self.mz_index = <double*>malloc(sizeof(double) * n)
        for i in range(n):
            p = self.getitem(i)
            self.mz_index[i] = p.mz

        if n > 2 and interval_index_size > 0:
        # if False:
            if self.interval_index != NULL:
                free_index_list(self.interval_index)
                self.interval_index = NULL
            interval_index = <index_list*>malloc(sizeof(index_list))
            build_interval_index(self, interval_index, interval_index_size)
            if check_index(interval_index) != 0:
                free_index_list(interval_index)
            else:
                self.interval_index = interval_index
        else:
            if self.interval_index != NULL:
                free_index_list(self.interval_index)
                self.interval_index = NULL

    cpdef size_t _index(self):
        i = PeakSet._index(self)
        self._allocate_index(INTERVAL_INDEX_SIZE)
        return i

    @cython.cdivision(True)
    cdef FittedPeak _has_peak(self, double mz, double tolerance=1e-5):
        cdef:
            size_t i, n, s
            FittedPeak peak
        n = self.get_size()
        if n == 0:
            return _null_peak
        if self.interval_index != NULL:
            find_search_interval(self.interval_index, mz, &s, &n)
            i = double_binary_search_ppm_with_hint(self.mz_index, mz, tolerance, n, s)
        else:
            i = double_binary_search_ppm(self.mz_index, mz, tolerance, n)
        peak = self.getitem(i)
        if abs((peak.mz - mz) / mz) < tolerance:
            return peak
        else:
            return _null_peak

    @cython.cdivision(True)
    def get_nearest_peak(self, double mz):
        cdef:
            double errout
            FittedPeak peak
        peak = self._get_nearest_peak(mz, &errout)
        return peak, errout

    @cython.cdivision(True)
    cdef FittedPeak _get_nearest_peak(self, double mz, double* errout):
        cdef:
            size_t i, n, s
            FittedPeak peak
        n = self.get_size()
        if n == 0:
            errout[0] = INF
            return _null_peak
        if self.interval_index != NULL:
            find_search_interval(self.interval_index, mz, &s, &n)
            i = double_binary_search_nearest_match_with_hint(self.mz_index, mz, n, s)
        else:
            i = double_binary_search_nearest_match(self.mz_index, mz, n)
        peak = self.getitem(i)
        errout[0] = (peak.mz - mz) / mz
        return peak

    cpdef FittedPeak has_peak_hinted(self, double mz, double tolerance=2e-5, size_t hint_start=0, size_t hint_end=-1):
        cdef:
            size_t i, n
            FittedPeak peak
        n = self.get_size()
        if hint_start > n :
            raise ValueError("Hinted start index %d cannot be greater than the set size %d" % (hint_start, n))
        if hint_end > n:
            raise ValueError("Hinted start index %d cannot be greater than the set size %d" % (hint_end, n))
        if n == 0:
            return _null_peak
        i = double_binary_search_ppm_with_hint(self.mz_index, mz, tolerance, hint_end, hint_start)
        peak = self.getitem(i)
        if abs((peak.mz - mz) / mz) < tolerance:
            return peak
        else:
            return _null_peak

    def test_interval(self, double value):
        cdef:
            int status
            size_t start, end
            size_t i
        if self.interval_index == NULL:
            return None, None, None
        status = find_search_interval(self.interval_index, value, &start, &end)
        return start, end, interpolate_index(self.interval_index, value)

    def find_interval_for(self, double value):
        if self.interval_index == NULL:
            return None
        return interpolate_index(self.interval_index, value)

    def check_interval(self, size_t i):
        cdef:
            index_cell cell
        if self.interval_index == NULL:
            return None
        cell = self.interval_index.index[i]
        return cell


cdef double* build_linear_spaced_array(double low, double high, size_t n):
    cdef:
        double* array
        double delta
        size_t i

    delta = (high - low) / (n - 1)

    array = <double*>malloc(sizeof(double) * n)

    for i in range(n):
        array[i] = low + i * delta
    return array


cdef void free_index_list(index_list* index):
    free(index.index)
    free(index)


cdef int check_index(index_list* index) nogil:
    if index.size == 0:
        return 1
    elif (index.high - index.low) == 0:
        return 2
    else:
        return 0


cdef int build_interval_index(PeakSet peaks, index_list* index, size_t index_size):
    cdef:
        double* linear_spacing
        double current_value, err, next_value
        size_t i, start_i, end_i, index_i, peaks_size
        FittedPeak peak
    peaks_size = peaks.get_size()
    if peaks_size > 0:
        index.low = peaks.getitem(0).mz
        index.high = peaks.getitem(peaks_size - 1).mz
    else:
        index.low = 0
        index.high = 1
    index.size = index_size
    linear_spacing = build_linear_spaced_array(
        index.low,
        index.high,
        index_size)

    index.index = <index_cell*>malloc(sizeof(index_cell) * index_size)

    for index_i in range(index_size):
        current_value = linear_spacing[index_i]
        peak = peaks._get_nearest_peak(current_value, &err)
        if peaks_size > 0:
            start_i = peak.peak_count
            if index_i > 0:
                start_i = index.index[index_i - 1].end - 1
            if index_i == index_size - 1:
                end_i = peaks_size - 1
            else:
                next_value = linear_spacing[index_i + 1]
                i = peak.peak_count
                while i < peaks_size:
                    peak = peaks.getitem(i)
                    if abs(current_value - peak.mz) > abs(next_value - peak.mz):
                        break
                    i += 1
                end_i = i
        else:
            start_i = 0
            end_i = 0
        index.index[index_i].center_value = current_value
        index.index[index_i].start = start_i
        index.index[index_i].end = end_i

    free(linear_spacing)
    return 0


cdef size_t interpolate_index(index_list* index, double value):
    cdef:
        double v
        size_t i
    v = (((value - index.low) / (index.high - index.low)) * index.size)
    i = <size_t>v
    return i


cdef int find_search_interval(index_list* index, double value, size_t* start, size_t* end):
    cdef:
        size_t i
    if value > index.high:
        i = index.size - 1
    elif value < index.low:
        i = 0
    else:
        i = interpolate_index(index, value)
    if i > 0:
        if i < index.size:
            start[0] = index.index[i - 1].start
        else:
            # if we're at index.size or greater, act as if we're at the last
            # cell of the index
            start[0] = index.index[index.size - 2].start
    else:
        # if somehow the index were negative, this could end badly.
        start[0] = index.index[i].start
    if i >= (index.size - 1):
        end[0] = index.index[index.size - 1].end + 1
    else:
        end[0] = index.index[i + 1].end + 1
    return 0
