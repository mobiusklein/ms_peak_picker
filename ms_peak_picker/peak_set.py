import operator
import numpy as np

from .utils import Base, ppm_error, range


class FittedPeak(Base):

    __slots__ = [
        "mz", "intensity", "signal_to_noise", "peak_count",
        "index", "full_width_at_half_max", "area",
        "left_width", "right_width"
    ]

    def __init__(self, mz, intensity, signal_to_noise, peak_count, index, full_width_at_half_max, area,
                 left_width=0, right_width=0):
        self.mz = mz
        self.intensity = intensity
        self.signal_to_noise = signal_to_noise
        self.peak_count = peak_count
        self.index = index
        self.full_width_at_half_max = full_width_at_half_max
        self.area = area
        self.left_width = left_width
        self.right_width = right_width

    def clone(self):
        return FittedPeak(self.mz, self.intensity, self.signal_to_noise,
                          self.peak_count, self.index, self.full_width_at_half_max,
                          self.area, self.left_width, self.right_width)

    def __reduce__(self):
        return self.__class__, (self.mz, self.intensity, self.signal_to_noise,
                                self.peak_count, self.index, self.full_width_at_half_max,
                                self.area, self.left_width, self.right_width)

    def __hash__(self):
        return hash((self.mz, self.intensity, self.signal_to_noise, self.full_width_at_half_max))

    def __eq__(self, other):
        if other is None:
            return False
        return (abs(self.mz - other.mz) < 1e-5) and (
            abs(self.intensity - other.intensity) < 1e-5) and (
            abs(self.signal_to_noise - other.signal_to_noise) < 1e-5) and (
            abs(self.full_width_at_half_max - other.full_width_at_half_max) < 1e-5)

    def __ne__(self, other):
        return not (self == other)


def _get_nearest_peak(peaklist, mz):
    lo = 0
    hi = len(peaklist)

    tol = 1

    def sweep(lo, hi):
        best_error = float('inf')
        best_index = None
        for i in range(hi - lo):
            i += lo
            v = peaklist[i].mz
            err = abs(v - mz)
            if err < best_error:
                best_error = err
                best_index = i
        return peaklist[best_index], best_error

    def binsearch(lo, hi):
        if (hi - lo) < 5:
            return sweep(lo, hi)
        else:
            mid = (hi + lo) / 2
            v = peaklist[mid].mz
            if abs(v - mz) < tol:
                return sweep(lo, hi)
            elif v > mz:
                return binsearch(lo, mid)
            else:
                return binsearch(mid, hi)
    return binsearch(lo, hi)


class PeakSet(Base):
    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return len(self.peaks)

    def _index(self):
        self.peaks = sorted(self.peaks, key=operator.attrgetter('mz'))
        i = 0
        for i, peak in enumerate(self.peaks):
            peak.peak_count = i
        return i

    def has_peak(self, mz, tolerance=1e-5):
        return binary_search(self.peaks, mz, tolerance)

    get_nearest_peak = _get_nearest_peak

    def __repr__(self):
        return "<PeakSet %d Peaks>" % (len(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PeakSet(self.peaks[item])
        return self.peaks[item]

    def clone(self):
        return PeakSet(p.clone() for p in self)

    def between(self, m1, m2, tolerance=1e-5):
        p1, _ = self.get_nearest_peak(m1)
        p2, _ = self.get_nearest_peak(m2)

        return self[p1.peak_count - 1:p2.peak_count + 1]


def _sweep_solution(array, value, lo, hi, tolerance, verbose=False):
    best_index = -1
    best_error = float('inf')
    best_intensity = 0
    for i in range(hi - lo):
        target = array[lo + i]
        error = ppm_error(value, target.mz)
        abs_error = abs(error)
        if abs_error < tolerance and (abs_error < best_error * 1.1) and (target.intensity > best_intensity):
            best_index = lo + i
            best_error = abs_error
            best_intensity = target.intensity
    if best_index == -1:
        return None
    else:
        return array[best_index]


def _binary_search(array, value, lo, hi, tolerance, verbose=False):
    if (hi - lo) < 5:
        return _sweep_solution(array, value, lo, hi, tolerance, verbose)
    else:
        mid = (hi + lo) / 2
        target = array[mid]
        target_value = target.mz
        error = ppm_error(value, target_value)

        if abs(error) <= tolerance:
            return _sweep_solution(array, value, max(mid - (mid if mid < 5 else 5), lo), min(
                mid + 5, hi), tolerance, verbose)
        elif target_value > value:
            return _binary_search(array, value, lo, mid, tolerance, verbose)
        elif target_value < value:
            return _binary_search(array, value, mid, hi, tolerance, verbose)
    raise Exception("No recursion found!")


def binary_search(array, value, tolerance=2e-5, verbose=False):
    size = len(array)
    if size == 0:
        return None
    return _binary_search(array, value, 0, size, tolerance, verbose)


try:
    _FittedPeak = FittedPeak
    _PeakSet = PeakSet
    _p_binary_search = binary_search
    from ._c.peak_set import FittedPeak, PeakSet
except ImportError:
    pass


def to_array(peak_set):
    array = np.zeros((len(peak_set), 6))
    array[:, 0] = [p.mz for p in peak_set]
    array[:, 1] = [p.intensity for p in peak_set]
    array[:, 2] = [p.signal_to_noise for p in peak_set]
    array[:, 3] = [p.full_width_at_half_max for p in peak_set]
    array[:, 4] = [p.index for p in peak_set]
    array[:, 5] = [p.peak_count for p in peak_set]
    return array
