import numpy as np
from .search import get_nearest
from .fft_patterson_charge_state import fft_patterson_charge_state
from .peak_statistics import peak_area


class PeakIndex(object):
    def __init__(self, mz_array, intensity_array, peaks):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.peaks = peaks

    def clone(self):
        return PeakIndex(np.array(self.mz_array), np.array(self.intensity_array), self.peaks.clone())

    def get_nearest(self, mz, index):
        return get_nearest(self.mz_array, mz, index)

    def get_nearest_peak(self, mz):
        return self.peaks.get_nearest_peak(mz)

    def slice(self, start, stop):
        return PeakIndex(self.mz_array[start:stop], self.intensity_array[start:stop], self.peaks)

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
        lo = self.get_nearest(peak.mz - peak.mz * peak.full_width_at_half_max, peak.index)
        hi = self.get_nearest(peak.mz + peak.mz * peak.full_width_at_half_max, peak.index)
        return peak_area(self.mz_array, self.intensity_array, lo, hi)

    def set_peaks(self, peaks):
        self.peaks = self.peaks.__class__(tuple(peaks))


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
from ms_peak_picker._c.peak_index import PeakIndex

