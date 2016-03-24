cimport numpy as np
import numpy as np

from .search cimport get_nearest
from ms_peak_picker._c.peak_set cimport PeakSet, FittedPeak
from .peak_statistics import peak_area
from ms_peak_picker.fft_patterson_charge_state import fft_patterson_charge_state

cdef class PeakIndex(object):

    def __init__(self, mz_array, intensity_array, peaks):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.peaks = peaks

    def clone(self):
        return PeakIndex(np.array(self.mz_array), np.array(self.intensity_array), self.peaks.clone())

    def get_nearest(self, double mz, size_t index):
        return get_nearest(self.mz_array, mz, index)

    def get_nearest_peak(self, double mz):
        cdef:
            double err
        return self.peaks._get_nearest_peak(mz, &err), err

    def slice(self, size_t start, size_t stop):
        return (PeakIndex(self.mz_array[start:stop], self.intensity_array[start:stop],
                self.peaks.between(self.mz_array[start], self.mz_array[stop] + 1)))

    def between(self, double start, double stop):
        return self.peaks._between(start, stop)

    def has_peak_within_tolerance(self, double mz, double tol):
        return has_peak_within_tolerance(self.peaks, mz, tol)

    def __getitem__(self, index):
        return self.peaks[index]

    def has_peak(self, double mz, double tolerance=2e-5):
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
        lo = self.get_nearest(peak.mz - peak.full_width_at_half_max, peak.index)
        hi = self.get_nearest(peak.mz + peak.full_width_at_half_max, peak.index)
        return peak_area(self.mz_array, self.intensity_array, lo, hi)

    def set_peaks(self, peaks):
        self.peaks = self.peaks.__class__(tuple(peaks))

    def points_along(self, peak):
        lo = self.get_nearest(peak.mz - peak.full_width_at_half_max, peak.index)
        hi = self.get_nearest(peak.mz + peak.full_width_at_half_max, peak.index)
        return self.mz_array[lo:hi], self.intensity_array[lo:hi]

    def __reduce__(self):
        return PeakIndex, (self.mz_array, self.intensity_array, self.peaks)


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
            mid = (hi + lo) / 2
            v = peaklist[mid].mz
            if abs(v - mz) < tol:
                return mid
            elif v > mz:
                return binsearch(lo, mid)
            else:
                return binsearch(mid, hi)
    return binsearch(lo, hi)
