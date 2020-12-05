cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from ms_peak_picker._c.peak_set cimport FittedPeak

from ms_peak_picker._c.peak_statistics cimport (
    find_signal_to_noise, find_left_width, find_right_width,
    quadratic_fit, lorentzian_fit, peak_area)

from ms_peak_picker._c.search cimport get_nearest_binary, get_nearest

from numpy.linalg import LinAlgError
import logging

logger = logging.getLogger("peak_picker")
info = logger.info
debug = logger.debug


cdef str CENTROID = 'centroid'
cdef str PROFILE = 'profile'


cdef dict fit_type_map = {
    "quadratic": PeakFit.quadratic,
    "gaussian": PeakFit.quadratic,
    "lorenztian": PeakFit.lorentzian,
    "lorentzian": PeakFit.lorentzian,
    "apex": PeakFit.apex,
    PeakFit.apex: PeakFit.apex,
    PeakFit.quadratic: PeakFit.quadratic,
    PeakFit.lorentzian: PeakFit.lorentzian
}


cdef dict peak_mode_map = {
    CENTROID: PeakMode.centroid,
    PROFILE: PeakMode.profile,
    PeakMode.profile: PeakMode.profile,
    PeakMode.centroid: PeakMode.centroid
}



@cython.boundscheck(False)
cpdef bint is_increasing(np.ndarray[cython.floating, ndim=1] mz_array):
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
    cdef:
        size_t i, n
        cython.floating a, b

    n = mz_array.shape[0]
    if n <= 1:
        return True
    with nogil:
        a = mz_array[0]
        for i in range(1, n):
            b = mz_array[i]
            if a >= b:
                return False
            a = b
        return True


@cython.final
@cython.freelist(1000)
cdef class PartialPeakFitState(object):

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

    cpdef int reset(self) nogil:
        """Resets all the data in the object to initial configuration
        """
        self.set = False
        self.left_width = -1
        self.right_width = -1
        self.full_width_at_half_max = -1
        self.signal_to_noise = -1
        return 0


cdef class PeakProcessor(object):

    def __init__(self, fit_type='quadratic', peak_mode=PROFILE, signal_to_noise_threshold=1, intensity_threshold=1,
                 threshold_data=False, verbose=False, integrate=True):
        if fit_type not in fit_type_map:
            raise ValueError("Unknown fit_type %r" % (fit_type,))
        else:
            fit_type = fit_type_map[fit_type]
        if peak_mode not in peak_mode_map:
            raise ValueError("Unknown peak_mode %r" % (peak_mode,))
        else:
            peak_mode = peak_mode_map[peak_mode]

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

    cpdef double get_signal_to_noise_threshold(self):
        return self._signal_to_noise_threshold

    cpdef object set_signal_to_noise_threshold(self, double signal_to_noise_threshold):
        self._signal_to_noise_threshold = signal_to_noise_threshold

        if self.threshold_data:
            if self.get_signal_to_noise_threshold() != 0:
                self.background_intensity = self.get_intensity_threshold() / self.get_signal_to_noise_threshold()
            else:
                self.background_intensity = 1.

    signal_to_noise_threshold = property(
        get_signal_to_noise_threshold, set_signal_to_noise_threshold)

    cpdef double get_intensity_threshold(self):
        return self._intensity_threshold

    cpdef object set_intensity_threshold(self, double intensity_threshold):
        self._intensity_threshold = intensity_threshold
        if self.threshold_data:
            if self.get_signal_to_noise_threshold() != 0:
                self.background_intensity = intensity_threshold / self.get_signal_to_noise_threshold()
            elif intensity_threshold != 0:
                self.background_intensity = intensity_threshold
            else:
                self.background_intensity = 1.

    intensity_threshold = property(
        get_intensity_threshold, set_intensity_threshold)

    def discover_peaks(self, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array, start_mz=None, stop_mz=None):
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
        cdef:
            double _start_mz, _stop_mz

        if start_mz is None:
            _start_mz = -1
        else:
            _start_mz = start_mz
        if stop_mz is None:
            _stop_mz = -1
        else:
            _stop_mz = stop_mz
        return self._discover_peaks(mz_array, intensity_array, _start_mz, _stop_mz)

    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cpdef size_t _discover_peaks(self, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array, double start_mz, double stop_mz):
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
        cdef:
            Py_ssize_t size, start_index, stop_index, ihigh, ilow, index
            list peak_data
            bint verbose, is_centroid
            double intensity_threshold, signal_to_noise_threshold, signal_to_noise
            double current_intensity, last_intensity, next_intensity, sum_intensity
            double low_intensity, high_intensity
            double full_width_at_half_max, current_mz, mz
            FittedPeak peak

        size = len(intensity_array) - 1

        if size < 1:
            return 0

        if start_mz <= -1:
            start_mz = mz_array[0]
        if stop_mz <= -1:
            stop_mz = mz_array[len(mz_array) - 1]

        peak_data = []

        verbose = self.verbose
        is_centroid = self.peak_mode == PeakMode.centroid

        intensity_threshold = self.intensity_threshold
        signal_to_noise_threshold = self.signal_to_noise_threshold

        start_index = get_nearest_binary(mz_array, start_mz, 0, size)
        stop_index = get_nearest_binary(mz_array, stop_mz, start_index, size)

        if start_index <= 0 and not is_centroid:
            start_index = 1
        elif start_index < 0 and is_centroid:
            start_index = 0
        if stop_index >= size - 1 and not is_centroid:
            stop_index = size - 1
        elif stop_index >= size and is_centroid:
            stop_index = size

        for index in range(start_index, stop_index + 1):
            self.partial_fit_state.reset()
            full_width_at_half_max = -1
            current_intensity = intensity_array[index]

            current_mz = mz_array[index]

            # If we are dealing with pre-centroided data, just walk down the array making each
            # point a FittedPeak.
            if is_centroid:
                if current_intensity <= 0:
                    continue
                mz = mz_array[index]
                signal_to_noise = current_intensity / (intensity_threshold or 1.0)
                full_width_at_half_max = 0.025

                if signal_to_noise > signal_to_noise_threshold:
                    peak = FittedPeak._create(
                        mz, current_intensity, signal_to_noise,
                        full_width_at_half_max, full_width_at_half_max / 2.,
                        full_width_at_half_max / 2., len(peak_data), index, area)
                    peak_data.append(peak)
            # Otherwise, carry out the peak finding and fitting procedure.
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
                            self.background_intensity

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

                            peak = FittedPeak._create(
                                fitted_mz, current_intensity, signal_to_noise,
                                full_width_at_half_max, self.partial_fit_state.left_width,
                                self.partial_fit_state.right_width, len(peak_data), index, area)
                            peak_data.append(peak)

                        # Move past adjacent equal-height peaks
                        incremented = False
                        while index < size and intensity_array[index] == current_intensity:
                            incremented = True
                            index += 1
                        if index > 0 and index < size and incremented:
                            index -= 1
        self.peak_data.extend(peak_data)
        return len(peak_data)

    cpdef double find_full_width_at_half_max(self, Py_ssize_t index, np.ndarray[cython.floating, ndim=1] mz_array,
                                             np.ndarray[cython.floating, ndim=1] intensity_array, double signal_to_noise):
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
        cdef:
            double left, right, fwhm
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

    cpdef double fit_peak(self, Py_ssize_t index, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array):
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
        if self.fit_type == PeakFit.apex:
            return mz_array[index]
        elif self.fit_type == PeakFit.quadratic:
            return quadratic_fit(mz_array, intensity_array, index)
        elif self.fit_type == PeakFit.lorentzian:
            full_width_at_half_max = self.find_full_width_at_half_max(
                index, mz_array, intensity_array,
                self.partial_fit_state.signal_to_noise)
            if full_width_at_half_max != 0:
                return lorentzian_fit(mz_array, intensity_array, index, full_width_at_half_max)
            return mz_array[index]

        return 0.0

    cpdef double area(self, np.ndarray[cython.floating] mz_array, np.ndarray[cython.floating] intensity_array, double mz,
                      double full_width_at_half_max, Py_ssize_t index):
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
