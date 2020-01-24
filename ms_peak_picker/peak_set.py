import operator
import numpy as np

from .utils import Base, ppm_error, range


class FittedPeak(Base):
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
        """Creates a deep copy of the peak

        Returns
        -------
        FittedPeak
        """
        return FittedPeak(self.mz, self.intensity, self.signal_to_noise,
                          self.peak_count, self.index, self.full_width_at_half_max,
                          self.area, self.left_width, self.right_width)

    def __reduce__(self):
        return self.__class__, (self.mz, self.intensity, self.signal_to_noise,
                                self.peak_count, self.index, self.full_width_at_half_max,
                                self.area, self.left_width, self.right_width)

    def __hash__(self):
        return hash(self.mz)

    def __eq__(self, other):
        if other is None:
            return False
        return (abs(self.mz - other.mz) < 1e-5) and (
            abs(self.intensity - other.intensity) < 1e-5) and (
            abs(self.signal_to_noise - other.signal_to_noise) < 1e-5) and (
            abs(self.full_width_at_half_max - other.full_width_at_half_max) < 1e-5)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return ("FittedPeak(mz=%0.3f, intensity=%0.3f, signal_to_noise=%0.3f, peak_count=%d, "
                "index=%d, full_width_at_half_max=%0.3f, area=%0.3f)") % (
                    self.mz, self.intensity, self.signal_to_noise,
                    self.peak_count, self.index, self.full_width_at_half_max,
                    self.area)


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
            mid = (hi + lo) // 2
            v = peaklist[mid].mz
            if abs(v - mz) < tol:
                return sweep(lo, hi)
            elif v > mz:
                return binsearch(lo, mid)
            else:
                return binsearch(mid, hi)
    return binsearch(lo, hi)


class PeakSet(Base):
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

    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return len(self.peaks)

    def reindex(self):
        """Re-indexes the sequence of peaks, updating their
        :attr:`peak_count` and setting their :attr:`index` if
        it is missing.
        """
        self._index()

    def _index(self):
        self.peaks = sorted(self.peaks, key=operator.attrgetter('mz'))
        i = 0
        for i, peak in enumerate(self.peaks):
            peak.peak_count = i
            if peak.index == -1:
                peak.index = i
        return i

    def has_peak(self, mz, tolerance=1e-5):
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
        return binary_search(self.peaks, mz, tolerance)

    get_nearest_peak = _get_nearest_peak

    def __repr__(self):
        return "<PeakSet %d Peaks>" % (len(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PeakSet(self.peaks[item])
        return self.peaks[item]

    def clone(self):
        """Creates a deep copy of the sequence of peaks

        Returns
        -------
        PeakSet
        """
        return PeakSet(p.clone() for p in self)

    def between(self, m1, m2):
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
        p1, _ = self.get_nearest_peak(m1)
        p2, _ = self.get_nearest_peak(m2)

        start = p1.peak_count
        end = p2.peak_count + 1
        start = p1.peak_count
        end = p2.peak_count
        n = len(self)
        if p1.mz < m1 and start + 1 < n:
            start += 1
        if p2.mz > m2 and end > 0:
            end -= 1
        return self[start:end]

    def all_peaks_for(self, mz, error_tolerance=2e-5):
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
        m1 = mz - (mz * error_tolerance)
        m2 = mz + (mz * error_tolerance)

        p1, _ = self.get_nearest_peak(m1)
        p2, _ = self.get_nearest_peak(m2)

        start = p1.peak_count
        end = p2.peak_count + 1
        start = p1.peak_count
        end = p2.peak_count
        n = len(self)
        if p1.mz < m1 and start + 1 < n:
            start += 1
        if p2.mz > m2 and end > 0:
            end -= 1
        return self.peaks[start:end]

    def __eq__(self, other):
        if other is None:
            return False
        return tuple(self) == tuple(other)

    def __ne__(self, other):
        return not (self == other)


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
        mid = (hi + lo) // 2
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
    _has_c = True
    _FittedPeak = FittedPeak
    _PeakSet = PeakSet
    _p_binary_search = binary_search
    from ._c.peak_set import FittedPeak, PeakSetIndexed as PeakSet
except ImportError:
    _has_c = False


def to_array(peak_set):
    array = np.zeros((len(peak_set), 6))
    array[:, 0] = [p.mz for p in peak_set]
    array[:, 1] = [p.intensity for p in peak_set]
    array[:, 2] = [p.signal_to_noise for p in peak_set]
    array[:, 3] = [p.full_width_at_half_max for p in peak_set]
    array[:, 4] = [p.index for p in peak_set]
    array[:, 5] = [p.peak_count for p in peak_set]
    return array


def simple_peak(mz, intensity, fwhm=0.01):
    return FittedPeak(mz, intensity, intensity, -1, -1,
                      fwhm, intensity, fwhm / 2, fwhm / 2)


def is_peak(obj):
    return hasattr(obj, 'mz') and hasattr(obj, 'intensity')
