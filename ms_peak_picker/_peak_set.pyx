from cpython.list cimport PyList_GET_ITEM
from cpython cimport PyObject

cdef double ppm_error(double x, double y):
    return (x - y) / y

cdef class FittedPeak(object):
    cdef:
        public double mz
        public double intensity
        public double signal_to_noise
        public double full_width_at_half_max
        public long peak_count
        public long index

    def __init__(self, mz, intensity, signal_to_noise, peak_count, index, full_width_at_half_max):
            self.mz = mz
            self.intensity = intensity
            self.signal_to_noise = signal_to_noise
            self.peak_count = peak_count
            self.index = index
            self.full_width_at_half_max = full_width_at_half_max

    def clone(self):
        return FittedPeak(self.mz, self.intensity, self.signal_to_noise,
                          self.peak_count, self.index, self.full_width_at_half_max)

    def __repr__(self):
        return ("FittedPeak(mz=%0.3f, intensity=%0.3f, signal_to_noise=%0.3f, peak_count=%d, "
                "index=%d, full_width_at_half_max=%0.3f") % (
                self.mz, self.intensity, self.signal_to_noise,
                self.peak_count, self.index, self.full_width_at_half_max)

    def __getstate__(self):
        return (self.mz, self.intensity, self.signal_to_noise,
                self.peak_count, self.index, self.full_width_at_half_max)

    def __setstate__(self, state):
        (self.mz, self.intensity, self.signal_to_noise,
         self.peak_count, self.index, self.full_width_at_half_max) = state

    def __reduce__(self):
        return FittedPeak, (0, 0, 0, 0, 0, 0), self.__getstate__()


cdef class PeakSet(object):
    cdef:
        public tuple peaks

    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return len(self.peaks)

    cpdef FittedPeak has_peak(self, double mz, double tolerance=1e-5):
        return binary_search(self.peaks, mz, tolerance)

    def __repr__(self):
        return "<PeakSet %d Peaks>" % (len(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PeakSet(self.peaks[item])
        return self.peaks[item]

    def clone(self):
        return PeakSet(p.clone() for p in self)

    def between(self, m1, m2, tolerance=1e-5):
        acc = []
        collecting = False
        for peak in self:
            if not collecting and peak.mz >= m1:
                collecting = True
            elif collecting and peak.mz >= m2:
                break
            elif collecting:
                acc.append(peak)
        return self.__class__(acc)

    def __getstate__(self):
        return self.tuples

    def __setstate__(self, state):
        self.peaks = state

    def __reduce__(self):
        return PeakSet, (tuple(),), self.__getstate__()


cpdef FittedPeak binary_search(list array, double value, double tolerance):
    return _binary_search(array, value, 0, len(array), tolerance)


cdef FittedPeak _binary_search(list array, double value, size_t lo, size_t hi, double tolerance):
    cdef:
        size_t mid
        FittedPeak target
        double target_value, error
    if (hi - lo) < 5:
        return _sweep_solution(array, value, lo, hi, tolerance)
    else:
        mid = (hi + lo) / 2
        target = <FittedPeak>PyList_GET_ITEM(array, mid)
        target_value = target.mz
        error = ppm_error(value, target_value)

        if abs(error) <= tolerance:
            return _sweep_solution(array, value, max(mid - 5, lo), min(mid + 5, hi), tolerance)
        elif target_value > value:
            return _binary_search(array, value, lo, mid, tolerance)
        elif target_value < value:
            return _binary_search(array, value, mid, hi, tolerance)

cdef FittedPeak _sweep_solution(list array, double value, size_t lo, size_t hi, double tolerance):
    cdef:
        long best_size
        double best_error, error, abs_error
        size_t i
    best_index = -1
    best_error = 1000000000000000
    for i in range(hi - lo):
        target = <FittedPeak>PyList_GET_ITEM(array, lo + i)
        error = ppm_error(value, target.mz)
        abs_error = abs(error)
        if abs_error < tolerance and abs_error < best_error:
            best_index = lo + i
            best_error = abs_error
    if best_index == -1:
        return None
    else:
        return <FittedPeak>PyList_GET_ITEM(array, best_index)