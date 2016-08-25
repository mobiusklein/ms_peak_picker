import numpy as np
from .search import get_nearest
from .fft_patterson_charge_state import fft_patterson_charge_state
from .peak_statistics import peak_area


class PeakIndex(object):
    def __init__(self, mz_array, intensity_array, peaks):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.peaks = peaks

    def pack(self):
        return PeakIndex(np.array([], dtype=np.float64), np.array([], dtype=np.float64), self.peaks.clone())

    def clone(self):
        return PeakIndex(np.array(self.mz_array), np.array(self.intensity_array), self.peaks.clone())

    def get_nearest(self, mz, index):
        if self.mz_array is None:
            raise ValueError("Cannot call get_nearest when raw arrays are None")

        return get_nearest(self.mz_array, mz, index)

    def get_nearest_peak(self, mz):
        return self.peaks.get_nearest_peak(mz)

    def slice(self, start, stop):
        return (PeakIndex(self.mz_array[start:stop], self.intensity_array[start:stop],
                self.peaks.between(self.mz_array[start], self.mz_array[stop] + 1)))

    def between(self, start, stop):
        return self.peaks.between(start, stop)

    def has_peak_within_tolerance(self, mz, tol):
        return has_peak_within_tolerance(self.peaks, mz, tol)

    def __getitem__(self, index):
        return self.peaks[index]

    def has_peak(self, mz, tolerance=2e-5):
        return self.peaks.has_peak(mz, tolerance)

    predict_charge_state = fft_patterson_charge_state

    def __len__(self):
        return len(self.peaks)

    def area(self, peak):
        if self.mz_array is None:
            raise ValueError("Cannot call area when raw arrays are None")

        lo = self.get_nearest(peak.mz - peak.full_width_at_half_max, peak.index)
        hi = self.get_nearest(peak.mz + peak.full_width_at_half_max, peak.index)
        return peak_area(self.mz_array, self.intensity_array, lo, hi)

    def points_along(self, peak, width=None):
        if self.mz_array is None:
            raise ValueError("Cannot call points_along when raw arrays are None")

        if width is None:
            width = peak.full_width_at_half_max
        lo = self.get_nearest(peak.mz - width, peak.index)
        hi = self.get_nearest(peak.mz + width, peak.index)
        return self.mz_array[lo:hi], self.intensity_array[lo:hi]

    def set_peaks(self, peaks):
        self.peaks = self.peaks.__class__(tuple(peaks))

    def __repr__(self):
        return "PeakIndex(%d points, %d peaks)" % (len(self.mz_array), len(self.peaks))


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


_PeakIndex = PeakIndex
try:
    from ms_peak_picker._c.peak_index import PeakIndex
except:
    pass
