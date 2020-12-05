'''
A Peak Picker/Fitter adapted from Decon2LS's DeconEngine
'''

import numpy as np
from numpy.linalg import LinAlgError

from .peak_statistics import (
    find_signal_to_noise, quadratic_fit, lorentzian_fit,
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
    "lorenztian": "lorentzian",
    "lorentzian": "lorentzian",
    "apex": "apex"
}


CENTROID = "centroid"
PROFILE = "profile"


peak_mode_map = {
    CENTROID: CENTROID,
    PROFILE: PROFILE
}


class PartialPeakFitState(Base):
    """Stores partial state for the peak currently being picked by a :class:`PeakProcessor`
    instance.

    Rather than storing this information directly in the :class:`PeakProcessor` object,
    this separates the state of the current peak from the state of the peak picking process,
    while providing a simple way to clear the current peak's data.

    Attributes
    ----------
    full_width_at_half_max : float
        The complete full width at half max of the current peak
    left_width : float
        The left width at half max of the current peak
    right_width : float
        The right width at half max of the current peak
    set : bool
        Whether or not the current peak has any stored data
    signal_to_noise : float
        The signal to noise ratio of the current peak
    """

    def __init__(self, left_width=-1, right_width=-1, full_width_at_half_max=-1, signal_to_noise=-1):
        self.set = left_width != -1
        self.left_width = left_width
        self.right_width = right_width
        self.full_width_at_half_max = full_width_at_half_max
        self.signal_to_noise = signal_to_noise

    def reset(self):
        """Resets all the data in the object to initial configuration
        """
        self.set = False
        self.left_width = -1
        self.right_width = -1
        self.full_width_at_half_max = -1
        self.signal_to_noise = -1


class PeakProcessor(object):
    """Directs the peak picking process, encapsulating the apex finding,
    peak fitting, and signal-to-noise estimation tasks.

    Attributes
    ----------
    background_intensity : float
        A static background intensity to use when estimating signal-to-noise
        ratio.
    fit_type : str
        The type of peak to fit
    intensity_threshold : float
        The minimum intensity required to accept a peak
    partial_fit_state : PartialPeakFitState
        A stateful container of measurements for the peak currently
        being fitted
    peak_data : list
        A list of :class:`ms_peak_picker.peak_set.FittedPeak` instances
    peak_mode : str
        Whether the peaks being picked are in profile mode or already centroided
        and just need to be passed directly into :class:`FittedPeak` instances
    signal_to_noise_threshold : float
        The minimum signal-to-noise ratio required to accept a peak fit
    threshold_data : bool
        Whether or not to enforce a signal-to-noise and intensity threhsold
    verbose : bool
        Whether to log additional diagnostic information
    """

    def __init__(self, fit_type='quadratic', peak_mode=PROFILE, signal_to_noise_threshold=1, intensity_threshold=1,
                 threshold_data=False, verbose=False, integrate=True):
        if fit_type not in fit_type_map:
            raise ValueError("Unknown fit_type %r" % (fit_type,))
        if peak_mode not in peak_mode_map:
            raise ValueError("Unknown peak_mode %r" % (peak_mode,))
        # normalize fit_type
        fit_type = fit_type_map[fit_type]
        self._signal_to_noise_threshold = 0
        self._intensity_threshold = 0

        self.fit_type = fit_type

        self.background_intensity = 1
        self.threshold_data = threshold_data
        self.signal_to_noise_threshold = signal_to_noise_threshold
        self.intensity_threshold = intensity_threshold

        self.peak_mode = peak_mode
        self.verbose = verbose
        self.integrate = integrate

        self.partial_fit_state = PartialPeakFitState()

        self.peak_data = []

    def get_signal_to_noise_threshold(self):
        return self._signal_to_noise_threshold

    def set_signal_to_noise_threshold(self, signal_to_noise_threshold):
        self._signal_to_noise_threshold = signal_to_noise_threshold

        if self.threshold_data:
            if self.signal_to_noise_threshold != 0:
                self.background_intensity = (
                    self.intensity_threshold / float(self.signal_to_noise_threshold))
            else:
                self.background_intensity = 1.

    signal_to_noise_threshold = property(
        get_signal_to_noise_threshold, set_signal_to_noise_threshold)

    def get_intensity_threshold(self):
        return self._intensity_threshold

    def set_intensity_threshold(self, intensity_threshold):
        self._intensity_threshold = intensity_threshold
        if self.threshold_data:
            if self.signal_to_noise_threshold != 0:
                self.background_intensity = intensity_threshold / \
                    float(self.signal_to_noise_threshold)
            elif intensity_threshold != 0:
                self.background_intensity = intensity_threshold
            else:
                self.background_intensity = 1.

    intensity_threshold = property(
        get_intensity_threshold, set_intensity_threshold)

    def discover_peaks(self, mz_array, intensity_array, start_mz=None, stop_mz=None):
        """Carries out the peak picking process on `mz_array` and `intensity_array`. All
        peaks picked are appended to :attr:`peak_data`.

        Parameters
        ----------
        mz_array : np.ndarray
            The m/z values to pick peaks from
        intensity_array : np.ndarray
            The intensity values to pick peaks from
        start_mz : float, optional
            The minimum m/z to pick peaks above
        stop_mz : float, optional
            The maximum m/z to pick peaks below

        Returns
        -------
        int
            The current number of peaks accumulated
        """
        size = len(intensity_array) - 1

        if size < 1:
            return 0

        if start_mz is None:
            start_mz = mz_array[0]
        if stop_mz is None:
            stop_mz = mz_array[len(mz_array) - 1]

        peak_data = []

        verbose = self.verbose

        intensity_threshold = self.intensity_threshold
        signal_to_noise_threshold = self.signal_to_noise_threshold

        start_index = get_nearest_binary(mz_array, start_mz, 0, size)
        stop_index = get_nearest_binary(mz_array, stop_mz, start_index, size)

        if start_index <= 0 and self.peak_mode != CENTROID:
            start_index = 1
        elif start_index < 0 and self.peak_mode == CENTROID:
            start_index = 0
        if stop_index >= size - 1 and self.peak_mode != CENTROID:
            stop_index = size - 1
        elif stop_index >= size and self.peak_mode == CENTROID:
            stop_index = size

        for index in range(start_index, stop_index + 1):
            self.partial_fit_state.reset()
            full_width_at_half_max = -1
            current_intensity = intensity_array[index]

            current_mz = mz_array[index]

            if self.peak_mode == CENTROID:
                if current_intensity <= 0:
                    continue
                mz = mz_array[index]
                signal_to_noise = current_intensity / intensity_threshold
                full_width_at_half_max = 0.025
                if signal_to_noise > signal_to_noise_threshold:
                    peak_data.append(FittedPeak(mz, current_intensity, signal_to_noise, len(
                        peak_data), index, full_width_at_half_max, current_intensity))
            else:
                last_intensity = intensity_array[index - 1]
                next_intensity = intensity_array[index + 1]

                # Three point peak picking. Check if the peak is greater than
                # both the previous and next points
                if (current_intensity >= last_intensity and current_intensity >= next_intensity and
                        current_intensity >= intensity_threshold):
                    signal_to_noise = 0.
                    if not self.threshold_data:
                        signal_to_noise = find_signal_to_noise(
                            current_intensity, intensity_array, index)
                    else:
                        signal_to_noise = current_intensity / \
                            float(self.background_intensity)

                    # Run Full-Width Half-Max algorithm to try to improve SNR
                    if signal_to_noise < signal_to_noise_threshold:
                        try:
                            full_width_at_half_max = self.find_full_width_at_half_max(
                                index,
                                mz_array, intensity_array, signal_to_noise)
                        except LinAlgError:
                            full_width_at_half_max = 0
                        if 0 < full_width_at_half_max < 0.5:
                            ilow = get_nearest_binary(
                                mz_array, current_mz - self.partial_fit_state.left_width, 0, index)
                            ihigh = get_nearest_binary(
                                mz_array, current_mz + self.partial_fit_state.right_width, index, stop_index)

                            low_intensity = intensity_array[ilow]
                            high_intensity = intensity_array[ihigh]

                            sum_intensity = low_intensity + high_intensity

                            if sum_intensity:
                                signal_to_noise = (
                                    2. * current_intensity) / sum_intensity
                            else:
                                signal_to_noise = 10.

                    self.partial_fit_state.signal_to_noise = signal_to_noise
                    # Found a putative peak, fit it
                    if signal_to_noise >= signal_to_noise_threshold:
                        fitted_mz = self.fit_peak(
                            index, mz_array, intensity_array)
                        if verbose:
                            debug(
                                "Considering peak at %d with fitted m/z %r", index, fitted_mz)
                        if full_width_at_half_max == -1:
                            try:
                                full_width_at_half_max = self.find_full_width_at_half_max(
                                    index,
                                    mz_array, intensity_array, signal_to_noise)
                            except LinAlgError:
                                full_width_at_half_max = 0

                        if full_width_at_half_max > 0:
                            if self.integrate:
                                area = self.area(
                                    mz_array, intensity_array, fitted_mz, full_width_at_half_max, index)
                            else:
                                area = current_intensity
                            if full_width_at_half_max > 1.:
                                full_width_at_half_max = 1.

                            if signal_to_noise > current_intensity:
                                signal_to_noise = current_intensity

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
        """Calculate full-width-at-half-max for a peak centered at `index` from
        `mz_array` and `intensity_array`, using the `signal_to_noise` to detect
        when to stop searching.

        This method will set attributes on :attr:`partial_fit_state`.

        Parameters
        ----------
        index : int
            The index of peak apex
        mz_array : np.ndarray
            The m/z array to search in
        intensity_array : np.ndarray
            The intensity array to search in
        signal_to_noise : float
            The signal-to-noise ratio for this peak

        Returns
        -------
        float
            The symmetric full-width-at-half-max
        """
        try:
            left = find_left_width(
                mz_array, intensity_array, index, signal_to_noise)
        except np.linalg.LinAlgError:
            left = 1e-7
        try:
            right = find_right_width(
                mz_array, intensity_array, index, signal_to_noise)
        except np.linalg.LinAlgError:
            right = 1e-7

        if left < 1e-6:
            left = right
        elif right < 1e-6:
            right = left
        if right < 1e-6 and left < 1e-6:
            left = right = 0.15
        fwhm = left + right
        self.partial_fit_state.left_width = left
        self.partial_fit_state.right_width = right
        self.partial_fit_state.full_width_at_half_max = fwhm

        return fwhm

    def fit_peak(self, index, mz_array, intensity_array):
        """Performs the peak shape fitting procedure.

        Parameters
        ----------
        index : int
            The index to start the peak fit from
        mz_array : np.ndarray
            The m/z array to search in
        intensity_array : np.ndarray
            The intensity array to search in

        Returns
        -------
        float
            m/z of the fitted peak center
        """
        if self.fit_type == "apex":
            return mz_array[index]
        elif self.fit_type == "quadratic":
            return quadratic_fit(mz_array, intensity_array, index)
        elif self.fit_type == "lorentzian":
            full_width_at_half_max = self.find_full_width_at_half_max(
                index, mz_array, intensity_array,
                self.partial_fit_state.signal_to_noise)
            if full_width_at_half_max != 0:
                return lorentzian_fit(mz_array, intensity_array, index, full_width_at_half_max)
            return mz_array[index]

        return 0.0

    def area(self, mz_array, intensity_array, mz, full_width_at_half_max, index):
        """Integrate the peak found at `index` with width `full_width_at_half_max`,
        centered at `mz`.

        Parameters
        ----------
        mz_array : np.ndarray
            The m/z array to search in
        intensity_array : np.ndarray
            The intensity array to search in
        mz : float
            The center m/z to start from
        full_width_at_half_max : float
            The width to use when extracting
            the range of points to integrate
        index : int
            The index to start the search from

        Returns
        -------
        float
            The integrated peak area
        """
        lo = get_nearest(mz_array, mz - full_width_at_half_max, index)
        hi = get_nearest(mz_array, mz + full_width_at_half_max, index)
        return peak_area(mz_array, intensity_array, lo, hi)

    def __iter__(self):
        for peak in self.peak_data:
            yield peak


def pick_peaks(mz_array, intensity_array, fit_type='quadratic', peak_mode=PROFILE,
               signal_to_noise_threshold=1., intensity_threshold=1., threshold_data=False,
               target_envelopes=None, transforms=None, verbose=False,
               start_mz=None, stop_mz=None, integrate=True):
    """Picks peaks for the given m/z, intensity array pair, producing a centroid-containing
    PeakIndex instance.

    Applies each :class:`.FilterBase` in `transforms` in order to
    `mz_array` and `intensity_array`.

    Creates an instance of :class:`.PeakProcessor` and configures it according to the parameters
    passed. If `target_envelopes` is set, each region is handled by :meth:`.PeakProcessor.discover_peaks`
    otherwise, :meth:`.PeakProcessor.discover_peaks` is invoked with `start_mz` and `stop_mz`.

    Produces a :class:`.PeakIndex`, a fast searchable collection of :class:`.FittedPeak` objects.

    Parameters
    ----------
    mz_array : np.ndarray
        An array of m/z measurements. Will by converted into np.float64
        values
    intensity_array : np.ndarray
        An array of intensity measurements. Will by converted into np.float64
        values
    fit_type : str, optional
        The name of the peak model to use. One of "quadratic", "gaussian", "lorentzian", or "apex"
    peak_mode : str, optional
        Whether peaks are in "profile" mode or are pre"centroid"ed
    signal_to_noise_threshold : float, optional
        Minimum signal-to-noise measurement to accept a peak
    intensity_threshold : float, optional
        Minimum intensity measurement to accept a peak
    threshold_data : bool, optional
        Whether to apply thresholds to the data
    target_envelopes : list, optional
        A sequence of (start m/z, end m/z) pairs, limiting peak picking to only those intervals
    transforms : list, optional
        A list of :class:`scan_filter.FilterBase` instances or callable that
        accepts (mz_array, intensity_array) and returns (mz_array, intensity_array) or
        `str` matching one of the premade names in `scan_filter.filter_register`
    verbose : bool, optional
        Whether to log extra information while picking peaks
    start_mz : float, optional
        A minimum m/z value to start picking peaks from
    stop_mz : None, optional
        A maximum m/z value to stop picking peaks after
    integrate: bool, optional
        Whether to integrate along each peak to calculate the area. Defaults
        to :const:`True`, but the area value for each peak is not usually used
        by downstream algorithms for consistency, so this expensive operation
        can be omitted.

    Returns
    -------
    :class:`PeakIndex`
        Contains all fitted peaks, as well as the transformed m/z and
        intensity arrays
    """
    if transforms is None:
        transforms = []

    mz_array = np.asanyarray(mz_array, dtype=np.float64)
    intensity_array = np.asanyarray(intensity_array, dtype=np.float64)

    if len(mz_array) != len(intensity_array):
        raise ValueError("The m/z array and intensity array must be the same size!")

    # make sure the m/z array is properly sorted
    if not is_increasing(mz_array):
        indexing = np.argsort(mz_array)
        mz_array = mz_array[indexing]
        intensity_array = intensity_array[indexing]

    mz_array, intensity_array = transform(
        mz_array, intensity_array, transforms)

    if len(mz_array) < 1:
        return PeakIndex(mz_array, intensity_array, PeakSet([]))

    processor = PeakProcessor(
        fit_type, peak_mode, signal_to_noise_threshold, intensity_threshold, threshold_data,
        verbose=verbose, integrate=integrate)
    if target_envelopes is None:
        processor.discover_peaks(
            mz_array, intensity_array,
            start_mz=start_mz, stop_mz=stop_mz)
    else:
        for start, stop in sorted(target_envelopes):
            processor.discover_peaks(
                mz_array, intensity_array, start_mz=start, stop_mz=stop)
    peaks = PeakSet(processor)
    peaks._index()
    return PeakIndex(mz_array, intensity_array, peaks)


def is_increasing(mz_array):
    """Test whether the values in `mz_array` are increasing.

    Occaisionally, the m/z array is not completely sorted. This should
    efficiently check if the array is indeed not in the correct order and
    a more expensive sort and re-index operation needs to be performed.

    Parameters
    ----------
    mz_array : :class:`np.ndarray`
        The array to test.

    Returns
    -------
    bool:
        Whether the array is strictly increasing or not.
    """
    return np.all(mz_array[1:] > mz_array[:-1])

try:
    _has_c = True
    _PartialPeakFitState = PartialPeakFitState
    _PeakProcessor = PeakProcessor
    from ms_peak_picker._c.peak_picker import PartialPeakFitState, PeakProcessor, is_increasing
except ImportError as err:
    print(err)
    _has_c = False
