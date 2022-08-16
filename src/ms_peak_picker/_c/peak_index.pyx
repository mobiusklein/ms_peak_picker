# cython: embedsignature=True


cimport numpy as np
import numpy as np

from .search cimport get_nearest
from ms_peak_picker._c.peak_set cimport PeakSet, FittedPeak
from .peak_statistics import peak_area
from ms_peak_picker.fft_patterson_charge_state import fft_patterson_charge_state

cdef class PeakIndex(object):
    """Represent a pair of m/z and intensity arrays paired with
    their :class:`PeakSet`.

    This class provides wrappers for most :class:`PeakSet` methods which
    it provides for its :attr:`peaks` attribute, and relates these peaks
    to their raw signal, making re-integration more convenient.

    This type also provides some limited multi-peak analysis
    to estimate the charge state of a peak using the isotopic pattern-free
    using Senko's Fourier Patterson Charge State determination algorithm
    :meth:`fft_patterson_charge_state`

    Attributes
    ----------
    mz_array : np.ndarray
        The original m/z array the peaks were picked from
    intensity_array : np.ndarray
        The original intensity array the peaks were picked from
    peaks : PeakSet
        The set of :class:`FittedPeak` objects picked from the associated
        arrays
    """
    def __init__(self, mz_array, intensity_array, peaks):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.peaks = peaks

    def pack(self):
        """Create a copy of `self` with the large
        arrays stripped out. This removes most functionality
        beyond wrapping the underlying :class:`PeakSet` but
        makes the object smaller.

        Returns
        -------
        PeakIndex
        """
        return PeakIndex(np.array([], dtype=np.float64), np.array([], dtype=np.float64), self.peaks.clone())

    cpdef PeakIndex clone(self):
        """Create a deep copy of `self`

        Returns
        -------
        PeakIndex
        """
        return PeakIndex(np.array(self.mz_array), np.array(self.intensity_array), self.peaks.clone())

    cpdef PeakIndex copy(self):
        return self.clone()

    def get_nearest(self, double mz, size_t index):
        """Get the nearest index to `mz` in the underlying arrays

        Parameters
        ----------
        mz : float
            The m/z to search for
        index : int
            The index to search from

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If the underlying arrays have been stripped, this method
            cannot be used
        """
        mz_array = self.mz_array
        if mz_array is None:
            raise ValueError("Cannot call get_nearest when raw arrays are None")
        elif mz_array.dtype == np.float64:
            return get_nearest(<np.ndarray[np.float64_t, ndim=1, mode='c']>mz_array, mz, index)
        elif mz_array.dtype == np.float32:
            return get_nearest(<np.ndarray[np.float32_t, ndim=1, mode='c']>mz_array, mz, index)
        else:
            return get_nearest(<np.ndarray[np.float64_t, ndim=1, mode='c']>(mz_array.astype(np.float64)), mz, index)

    def get_nearest_peak(self, double mz):
        """Wraps :meth:`PeakSet.get_nearest_peak`

        Parameters
        ----------
        mz : float
            The m/z to search with

        Returns
        -------
        tuple of (FittedPeak, float)
            The nearest peak and the m/z delta between that peak and `mz`
        """
        cdef:
            double err
        return self.peaks._get_nearest_peak(mz, &err), err

    def slice(self, size_t start, size_t stop):
        return (PeakIndex(self.mz_array[start:stop], self.intensity_array[start:stop],
                self.peaks.between(self.mz_array[start], self.mz_array[stop] + 1)))

    def between(self, double start, double stop):
        """Wraps :meth:`PeakSet.between`

        Parameters
        ----------
        start : float
            The lower m/z limit
        stop : float
            The upper m/z limit

        Returns
        -------
        PeakSet
        """
        return self.peaks._between(start, stop)

    cdef PeakSet _between(self, double start, double stop):
        return self.peaks._between(start, stop)

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
        return self.peaks.all_peaks_for(mz, error_tolerance)

    cdef size_t get_size(self):
        return self.peaks.get_size()

    def has_peak_within_tolerance(self, double mz, double tol):
        return has_peak_within_tolerance(self.peaks, mz, tol)

    def __getitem__(self, index):
        return self.peaks[index]

    def __eq__(self, other):
        return self.peaks == other.peaks

    def __ne__(self, other):
        return self.peaks != other.peaks

    def has_peak(self, double mz, double tolerance=2e-5):
        """Wraps :meth:`PeakSet.has_peak`

        Parameters
        ----------
        mz : float
            The m/z to search for
        tolerance : float, optional
            The error tolerance to accept. Defaults to 2e-5 (20 ppm)

        Returns
        -------
        FittedPeak
            The peak, if found. `None` otherwise.
        """
        cdef:
            FittedPeak peak
        peak = self.peaks._has_peak(mz, tolerance)
        if peak.mz == -1:
            return None
        return peak

    cdef FittedPeak _has_peak(self, double mz, double tolerance=2e-5):
        cdef:
            FittedPeak peak
        peak = self.peaks._has_peak(mz, tolerance)
        if peak.mz == -1:
            return None
        return peak

    predict_charge_state = fft_patterson_charge_state

    def __len__(self):
        return len(self.peaks)

    def area(self, peak):
        if self.mz_array is None:
            raise ValueError("Cannot call area when raw arrays are None")

        lo = self.get_nearest(peak.mz - peak.full_width_at_half_max, peak.index)
        hi = self.get_nearest(peak.mz + peak.full_width_at_half_max, peak.index)
        return peak_area(self.mz_array, self.intensity_array, lo, hi)

    def set_peaks(self, peaks):
        self.peaks = self.peaks.__class__(tuple(peaks))

    def points_along(self, peak, width=None):
        if self.mz_array is None:
            raise ValueError("Cannot call points_along when raw arrays are None")

        if width is None:
            width = peak.full_width_at_half_max
        lo = self.get_nearest(peak.mz - width, peak.index)
        hi = self.get_nearest(peak.mz + width, peak.index)
        return self.mz_array[lo:hi], self.intensity_array[lo:hi]

    def __reduce__(self):
        return PeakIndex, (self.mz_array, self.intensity_array, self.peaks)

    def __repr__(self):
        return "PeakIndex(%d points, %d peaks)" % (len(self.mz_array), len(self.peaks))

cdef int _has_peak_within_tolerance(PeakSet peaklist, double mz, double tol, size_t *out):
    cdef:
        size_t lo, hi, mid, i
        FittedPeak peak
        double v

    lo = 0
    hi = len(peaklist)

    while hi != lo:
        mid = (hi + lo) // 2
        peak = peaklist.getitem(mid)
        v = peak.mz
        if abs(v - mz) < tol:
            out[0] = mid
            return True
        elif (hi - lo) == 1:
            return False
        elif v > mz:
            hi = mid
        else:
            lo = mid
    return False


def chas_peak_within_tolerance(PeakSet peaklist, double mz, double tol):
    cdef:
        size_t out

    if _has_peak_within_tolerance(peaklist, mz, tol, &out):
        return out
    return False


def has_peak_within_tolerance(peaklist, mz, tol):
    lo = 0
    hi = len(peaklist)

    def sweep(lo, hi):
        for i in range(hi - lo):
            i += lo
            v = peaklist[i].mz
            if abs(v - mz) < tol:
                return i or True
        return False

    def binsearch(lo, hi):
        if (hi - lo) < 5:
            return sweep(lo, hi)
        else:
            mid = (hi + lo) // 2
            v = peaklist[mid].mz
            if abs(v - mz) < tol:
                return mid
            elif v > mz:
                return binsearch(lo, mid)
            else:
                return binsearch(mid, hi)
    return binsearch(lo, hi)
