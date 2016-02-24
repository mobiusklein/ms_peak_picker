from .utils import Base, ppm_error, range


class FittedPeak(Base):
    def __init__(self, mz, intensity, signal_to_noise, peak_count, index, full_width_at_half_max):
        self.mz = mz
        self.intensity = intensity
        self.signal_to_noise = signal_to_noise
        self.peak_count = peak_count
        self.index = index
        self.full_width_at_half_max = full_width_at_half_max


class PeakSet(Base):
    def __init__(self, peaks):
        self.peaks = tuple(peaks)

    def __len__(self):
        return len(self.peaks)

    def has_peak(self, mz, tolerance=1e-5):
        return binary_search(self.peaks, mz, tolerance)

    def __repr__(self):
        return "<PeakSet %d Peaks>" % (len(self))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PeakSet(self.peaks[item])
        return self.peaks[item]


def _sweep_solution(array, value, lo, hi, tolerance, verbose=False):
    best_index = -1
    best_error = float('inf')
    for i in range(hi - lo):
        target = array[lo + i]
        error = ppm_error(value, target.mz)
        abs_error = abs(error)
        if abs_error < tolerance and abs_error < best_error:
            best_index = lo + i
            best_error = abs_error
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
            return _sweep_solution(array, value, max(mid - 5, lo), min(mid + 5, hi), tolerance, verbose)
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
