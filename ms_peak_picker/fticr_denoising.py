'''
An implementation of the noise filtering methods from MasSpike
'''

import numpy as np


def binsearch(array, x, hint=None):
    n = len(array)
    lo = 0
    hi = n

    while hi != lo:
        mid = (hi + lo) // 2
        y = array[mid]
        err = y - x
        if hi - lo == 1:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


def between_search(array, lo, hi):
    n = len(array)
    lo_i = binsearch(array, lo)
    hi_i = binsearch(array, hi)

    if lo - array[lo_i] > 0.1:
        if lo_i < n - 1:
            lo_i += 1
    if ((array[hi_i] - hi) > 0.1) and (hi_i - 1) > lo_i:
        if hi_i > 0:
            hi_i -= 1
    if lo_i > hi_i:
        hi_i = lo_i
    return lo_i, hi_i


class Window(object):

    def __init__(self, mz_array, intensity_array, start_index=None, end_index=None,
                 center_mz=None):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.start_index = start_index
        self.end_index = end_index
        self.center_mz = center_mz

    def deduct_intensity(self, value):
        self.intensity_array -= value
        self.intensity_array.clip(min=0, out=self.intensity_array)

    @property
    def mean_intensity(self):
        return self.intensity_array.mean()

    def truncated_mean(self, threshold=0.95):
        try:
            hist_count, hist_values = np.histogram(self.intensity_array)
        except ValueError:
            return 1e-6
        mask = (hist_count.max() * (1 - threshold)) < hist_count
        mean = (hist_values[1:][mask] * hist_count[mask]
                ).sum() / hist_count[mask].sum()
        return mean

    def __repr__(self):
        return "Window(start_index=%d, end_index=%d, mean_intensity=%f, center_mz=%f)" % (
            self.start_index, self.end_index, self.mean_intensity, self.center_mz)


def windowed_spectrum(mz_array, intensity_array, window_size=1.):
    mz_min = mz_array.min()
    mz_max = mz_array.max()

    step_size = window_size / 2.
    center_mz = mz_min + step_size
    center_i = binsearch(mz_array, center_mz, 1)

    windows = []

    niter = 0

    while center_mz < mz_max:
        lo_mz = center_mz - step_size
        hi_mz = center_mz + step_size

        # lo_i = binsearch(mz_array, lo_mz, center_i)
        # hi_i = binsearch(mz_array, hi_mz, center_i)
        lo_i, hi_i = between_search(mz_array, lo_mz, hi_mz)
        mz_window = mz_array[lo_i:hi_i + 1]
        if lo_mz < mz_window.mean() < hi_mz:
            win = Window(mz_window,
                         intensity_array[lo_i:hi_i + 1], lo_i, hi_i)
        else:
            win = Window(np.array([center_mz]), np.array([0.0]), lo_i, hi_i)
        win.center_mz = center_mz
        windows.append(win)

        center_mz = center_mz + window_size
        center_i = binsearch(mz_array, center_mz, center_i)

        niter += 1

    return windows


class NoiseRegion(object):

    def __init__(self, windows, width=10):
        self.windows = windows
        self.width = width
        self.start_index = windows[0].start_index
        self.end_index = windows[-1].end_index

    def __repr__(self):
        return "NoiseRegion(start_index=%d, end_index=%d)" % (
            self.start_index, self.end_index)

    def noise_window(self):
        return min(self.windows, key=lambda x: x.mean_intensity)

    def arrays(self):
        mz_array = np.concatenate([w.mz_array for w in self.windows])
        intensity_array = np.concatenate(
            [w.intensity_array for w in self.windows])
        return mz_array, intensity_array

    def denoise(self, scale=5, maxiter=10):
        if scale == 0:
            return 0
        noise_mean = self.noise_window().truncated_mean() * scale
        for window in self.windows:
            window.deduct_intensity(noise_mean)
        last_mean = noise_mean
        noise_mean = self.noise_window().truncated_mean()
        niter = 1
        while abs(last_mean - noise_mean) > 1e-3 and niter < maxiter:
            niter += 1
            noise_mean = self.noise_window().truncated_mean() * scale
            for window in self.windows:
                window.deduct_intensity(noise_mean)
            last_mean = noise_mean

        return last_mean - noise_mean


def group_windows(windows, width=10):
    step = int(width / 2)
    regions = []
    i = step
    while i < len(windows):
        lo = i - step
        hi = i + step
        reg = NoiseRegion(windows[lo:hi])
        regions.append(reg)
        i += step * 2
    return regions


class FTICRScan(object):

    def __init__(self, mz_array, intensity_array):
        self.mz_array = mz_array.copy()
        self.intensity_array = intensity_array.copy()

        self.windows = None
        self.regions = None

    def __iter__(self):
        yield self.mz_array
        yield self.intensity_array

    def build_windows(self, window_size=1.):
        self.windows = windowed_spectrum(
            self.mz_array, self.intensity_array, window_size=window_size)

    def build_regions(self, region_width=10):
        self.regions = group_windows(self.windows, region_width)

    def denoise(self, window_size=1., region_width=10, scale=5):
        self.build_windows(window_size)
        self.build_regions(region_width)
        for region in self.regions:
            region.denoise(scale)
        return self


try:
    _FTICRScan = FTICRScan
    has_c = True
    from ms_peak_picker._c.fticr_denoising import FTICRScan
except ImportError:
    has_c = False


def denoise(mz_array, intensity_array, window_size=1., region_width=10, scale=5):
    scan = FTICRScan(mz_array, intensity_array)
    denoised = scan.denoise(window_size, region_width, scale)
    x, y = list(denoised)
    assert x.shape == mz_array.shape and y.shape == intensity_array.shape
    return x, y
