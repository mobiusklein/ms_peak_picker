'''
A Peak Picker/Fitter adapted from Decon2LS's DeconEngine
'''

from .peak_statistics import (
    find_signal_to_noise, quadratic_fit, lorenztian_fit,
    peak_area, find_left_width, find_right_width)

from .search import get_nearest_binary, get_nearest
from .peak_set import FittedPeak, PeakSet
from .peak_index import PeakIndex
from .scan_filter import transform
from .utils import Base


import logging

logger = logging.getLogger("peak_picker")
info = logger.info
debug = logger.debug


fit_type_map = {
    "quadratic": "quadratic",
    "gaussian": "quadratic",
    "lorenztian": "lorenztian",
}


class PartialPeakFitState(Base):
    def __init__(self, left_width=-1, right_width=-1, full_width_at_half_max=-1, signal_to_noise=-1):
        self.set = left_width != -1
        self.left_width = left_width
        self.right_width = right_width
        self.full_width_at_half_max = full_width_at_half_max
        self.signal_to_noise = signal_to_noise

    def reset(self):
        self.set = False
        self.left_width = -1
        self.right_width = -1
        self.full_width_at_half_max = -1
        self.signal_to_noise = -1


class PeakProcessor(object):
    _signal_to_noise_threshold = 0
    _intensity_threshold = 0

    def __init__(self, fit_type='quadratic', peak_mode='profile', signal_to_noise_threshold=1, intensity_threshold=1,
                 threshold_data=False, verbose=False):
        assert fit_type in fit_type_map
        self.fit_type = fit_type
        self.background_intensity = 1
        self.threshold_data = threshold_data
        self.signal_to_noise_threshold = signal_to_noise_threshold
        self.intensity_threshold = intensity_threshold
        self.peak_mode = peak_mode
        self.verbose = verbose

        self.partial_fit_state = PartialPeakFitState()

        self.peak_data = []

    def get_signal_to_noise_threshold(self):
        return self._signal_to_noise_threshold

    def set_signal_to_noise_threshold(self, signal_to_noise_threshold):
        self._signal_to_noise_threshold = signal_to_noise_threshold

        if self.threshold_data:
            if self.signal_to_noise_threshold != 0:
                self.background_intensity = (self.intensity_threshold / float(self.signal_to_noise_threshold))
            else:
                self.background_intensity = 1.

    signal_to_noise_threshold = property(get_signal_to_noise_threshold, set_signal_to_noise_threshold)

    def get_intensity_threshold(self):
        return self._intensity_threshold

    def set_intensity_threshold(self, intensity_threshold):
        self._intensity_threshold = intensity_threshold
        if self.threshold_data:
            if self.signal_to_noise_threshold != 0:
                self.background_intensity = intensity_threshold / float(self.signal_to_noise_threshold)
            elif intensity_threshold != 0:
                self.background_intensity = intensity_threshold
            else:
                self.background_intensity = 1.

    intensity_threshold = property(get_intensity_threshold, set_intensity_threshold)

    def discover_peaks(self, mz_array, intensity_array, start_mz=None, stop_mz=None):
        if start_mz is None:
            start_mz = mz_array[0]
        if stop_mz is None:
            stop_mz = mz_array[len(mz_array) - 1]

        if len(intensity_array) < 1:
            return 0
        peak_data = []
        size = len(intensity_array) - 1

        verbose = self.verbose

        intensity_threshold = self.intensity_threshold
        signal_to_noise_threshold = self.signal_to_noise_threshold

        start_index = get_nearest_binary(mz_array, start_mz, 0, size)
        stop_index = get_nearest_binary(mz_array, stop_mz, start_index, size)

        if start_index <= 0:
            start_index = 1
        if stop_index >= size - 1:
            stop_index = size - 1

        for index in range(start_index, stop_index + 1):
            self.partial_fit_state.reset()
            full_width_at_half_max = -1
            current_intensity = intensity_array[index]
            last_intensity = intensity_array[index - 1]
            next_intensity = intensity_array[index + 1]

            current_mz = mz_array[index]

            if self.peak_mode == "centroid":
                mz = mz_array[index]
                signal_to_noise = current_intensity / intensity_threshold
                full_width_at_half_max = 0.3
                peak_data.append(FittedPeak(mz, current_intensity, signal_to_noise, len(
                    peak_data), index, full_width_at_half_max, current_intensity))
            else:
                # Three point peak picking. Check if the peak is greater than both the previous and next points
                if (current_intensity >= last_intensity and current_intensity >= next_intensity and
                        current_intensity >= intensity_threshold):
                    signal_to_noise = 0.
                    if not self.threshold_data:
                        signal_to_noise = find_signal_to_noise(current_intensity, mz_array, index)
                    else:
                        signal_to_noise = current_intensity / float(self.background_intensity)

                    # Run Full-Width Half-Max algorithm to try to improve SNR
                    if signal_to_noise < signal_to_noise_threshold:
                        full_width_at_half_max = self.find_full_width_at_half_max(
                            index,
                            mz_array, intensity_array, signal_to_noise)
                        if 0 < full_width_at_half_max < 0.5:
                            ilow = get_nearest_binary(
                                mz_array, current_mz - self.partial_fit_state.left_width, 0, index)
                            ihigh = get_nearest_binary(
                                mz_array, current_mz + self.partial_fit_state.right_width, index, stop_index)

                            low_intensity = intensity_array[ilow]
                            high_intensity = intensity_array[ihigh]

                            sum_intensity = low_intensity + high_intensity

                            if sum_intensity:
                                signal_to_noise = (2. * current_intensity) / sum_intensity
                            else:
                                signal_to_noise = 10.

                    self.partial_fit_state.signal_to_noise = signal_to_noise
                    # Found a putative peak, fit it
                    if signal_to_noise >= signal_to_noise_threshold:
                        fitted_mz = self.fit_peak(index, mz_array, intensity_array)
                        if verbose:
                            debug("Considering peak at %d with fitted m/z %r", index, fitted_mz)
                        if full_width_at_half_max == -1:
                            full_width_at_half_max = self.find_full_width_at_half_max(
                                index,
                                mz_array, intensity_array, signal_to_noise)

                        if full_width_at_half_max > 0:
                            area = self.area(mz_array, intensity_array, fitted_mz, full_width_at_half_max, index)

                            peak_data.append(FittedPeak(
                                fitted_mz, current_intensity, signal_to_noise,
                                len(peak_data), index, full_width_at_half_max, area,
                                self.partial_fit_state.left_width, self.partial_fit_state.right_width))

                        # Move past adjacent equal-height peaks
                        incremented = False
                        while index < size and intensity_array[index] == current_intensity:
                            incremented = True
                            index += 1
                        if index > 0 and index < size and incremented:
                            index -= 1
        self.peak_data.extend(peak_data)
        return len(peak_data)

    def find_full_width_at_half_max(self, index, mz_array, intensity_array, signal_to_noise):
        left = find_left_width(mz_array, intensity_array, index, signal_to_noise)
        right = find_right_width(mz_array, intensity_array, index, signal_to_noise)

        if left < 1e-6:
            left = right
        elif right < 1e-6:
            right = left
        elif right < 1e-6 and left < 1e-6:
            left = right = 0.15
        fwhm = left + right
        self.partial_fit_state.left_width = left
        self.partial_fit_state.right_width = right
        self.partial_fit_state.full_width_at_half_max = fwhm

        return fwhm

    def fit_peak(self, index, mz_array, intensity_array):
        if self.fit_type == "apex":
            return mz_array[index]
        elif self.fit_type == "quadratic":
            return quadratic_fit(mz_array, intensity_array, index)
        elif self.fit_type == "lorenztian":
            full_width_at_half_max = self.find_full_width_at_half_max(
                index, mz_array, intensity_array,
                self.partial_fit_state.signal_to_noise)
            if full_width_at_half_max != 0:
                return lorenztian_fit(mz_array, intensity_array, index, full_width_at_half_max)
            return mz_array[index]

        return 0.0

    def area(self, mz_array, intensity_array, mz, full_width_at_half_max, index):
        lo = get_nearest(mz_array, mz - full_width_at_half_max, index)
        hi = get_nearest(mz_array, mz + full_width_at_half_max, index)
        return peak_area(mz_array, intensity_array, lo, hi)

    def __iter__(self):
        for peak in self.peak_data:
            yield peak


def pick_peaks(mz_array, intensity_array, fit_type='quadratic', peak_mode='profile',
               signal_to_noise_threshold=1, intensity_threshold=1, threshold_data=False,
               target_envelopes=None, transforms=None, verbose=False):
    if transforms is None:
        transforms = []

    mz_array = mz_array.astype(float)
    intensity_array = intensity_array.astype(float)

    mz_array, intensity_array = transform(mz_array, intensity_array, transforms)

    processor = PeakProcessor(
        fit_type, peak_mode, signal_to_noise_threshold, intensity_threshold, threshold_data,
        verbose=verbose)
    if target_envelopes is None:
        processor.discover_peaks(mz_array, intensity_array)
    else:
        for start, stop in sorted(target_envelopes):
            processor.discover_peaks(mz_array, intensity_array, start_mz=start, stop_mz=stop)
    peaks = PeakSet(processor)
    peaks._index()
    return PeakIndex(mz_array, intensity_array, peaks)
