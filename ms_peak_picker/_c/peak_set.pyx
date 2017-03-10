# cython: embedsignature=True
cimport cython
import operator

from cpython.tuple cimport PyTuple_GET_ITEM, PyTuple_GetItem, PyTuple_GetSlice, PyTuple_GET_SIZE
from cpython cimport PyObject

cdef double ppm_error(double x, double y):
    return (x - y) / y


@cython.freelist(100000)
cdef class FittedPeak(object):
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

    cpdef FittedPeak clone(self):
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
        return hash((self.mz, self.intensity, self.signal_to_noise, self.full_width_at_half_max))

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

    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return PyTuple_GET_SIZE(self.peaks)

    cdef size_t _get_size(self):
        return PyTuple_GET_SIZE(self.peaks)

    def reindex(self):
        self._index()

    def _index(self):
        self.peaks = tuple(sorted(self.peaks, key=operator.attrgetter('mz')))
        i = 0
        for i, peak in enumerate(self.peaks):
            peak.peak_count = i
            if peak.index == -1:
                peak.index = i
        return i

    def has_peak(self, double mz, double tolerance=1e-5):
        peak = self._has_peak(mz, tolerance)
        if peak is _null_peak:
            return None
        return peak

    def get_nearest_peak(self, double mz):
        cdef:
            FittedPeak peak
            double err
        peak = _binary_search_nearest_match(self.peaks, mz, 0, PyTuple_GET_SIZE(self.peaks), &err)
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

    def clone(self):
        return PeakSet(p.clone() for p in self)

    def between(self, double m1, double m2):
        cdef:
            FittedPeak p1
            FittedPeak p2
            double err
        p1 = self._get_nearest_peak(m1, &err)
        p2 = self._get_nearest_peak(m2, &err)

        return self[p1.peak_count - 1:p2.peak_count + 1]

    cdef PeakSet _between(self, double m1, double m2):
        cdef:
            FittedPeak p1
            FittedPeak p2
            double err
        p1 = self._get_nearest_peak(m1, &err)
        p2 = self._get_nearest_peak(m2, &err)
        return PeakSet(<tuple>PyTuple_GetSlice(self.peaks, p1.peak_count, p2.peak_count + 1))

    def __getstate__(self):
        return self.peaks

    def __setstate__(self, state):
        self.peaks = state

    def __reduce__(self):
        return PeakSet, (tuple(),), self.__getstate__()


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
        mid = (hi + lo) / 2
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
        mid = (hi + lo) / 2
        v = array[mid].mz
        if abs(v - value) < 1.:
            return _sweep_nearest_match(array, value, lo, hi, errout)
        elif v > value:
            return _binary_search_nearest_match(array, value, lo, mid, errout)
        else:
            return _binary_search_nearest_match(array, value, mid, hi, errout)

