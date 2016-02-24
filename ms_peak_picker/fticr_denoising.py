'''
An implementation of the noise filtering methods from MasSpike
'''

import numpy as np

from .utils import Base


class Scan(Base):
    def __init__(self, scan_id, signal, ms_level, precursor=None, **kwargs):
        self.scan_id = scan_id
        self.signal = signal
        self.ms_level = ms_level
        self._options = kwargs
        self.precursor = precursor

        self.noise_regions = []


class Window(Base):
    def __init__(self, signal_window, start, end, signal_to_noise):
        self.signal = signal_window
        self.start = start
        self.end = end
        self.signal_to_noise = signal_to_noise

    def spans(self, other):
        return (((self.start <= other.start) & (other.end < self.end)) |
                ((self.start >= other.start) & (other.end > self.start)))


class NoiseRegion(Base):
    def __init__(self, start, end, mean_noise, signal=None):
        self.start = start
        self.end = end
        self.mean_noise = mean_noise
        self.signal = signal[start:end]

    def spans(self, other):
        return (((self.start <= other.start) & (other.end < self.end)) |
                ((self.start >= other.start) & (other.end > self.start)))

    def __getitem__(self, idx):
        return self.signal[idx]

    def __setitem__(self, idx, value):
        self.signal[idx] = value

    def slice(self):
        return slice(self.start, self.end)


def pyteomics_converter(scan):
    mz = scan['m/z array'].astype(float)
    intensity = scan['intensity array'].astype(float)
    ms_level = scan['ms level']
    scan_id = scan['id']
    kwargs = {
        "title": scan.get('spectrum title') or scan.get('id'),
        "scan_list": scan["scanList"]
    }

    if ms_level > 1:
        pass

    mz.shape = (-1, 1)
    intensity.shape = (-1, 1)

    signal = np.concatenate((mz, intensity), axis=1)

    raw_scan = Scan(scan_id, signal, ms_level, **kwargs)
    return raw_scan


def noise_regions(signal, start=0, width=0.5, window_size=10):
    means = []
    regions = []
    i = 1
    last = 0
    base = signal[0, 0]

    window_size /= width
    window_size += 1
    window_size = int(window_size)

    while i < window_size:
        lo = base + (width * (start + i)) + -width
        hi = base + (width * (start + i)) + width
        mask = (signal[:, 0] >= lo) & (signal[:, 0] <= hi)
        window_mean = signal[mask, 1].mean()
        if np.isnan(window_mean):
            i += 1
            continue
        means.append(window_mean)
        regions.append(np.where(mask))
        i += 1
    return np.array(means), regions


def window_bounds(window):
    return window[1][0][0][0], window[1][-1][-1][-1]


def noise_window(signal, noise_regions_):
    idx = np.nanargmin(noise_regions_[0])
    noise_window = signal[noise_regions_[1][idx]][:, 1]
    noise_window = noise_window[noise_window <= noise_window.max() * 0.95]
    return noise_window.mean()


def denoise_region(signal, start=0, width=0.5):
    n = 10
    i = 1
    noise_regions_ = noise_regions(signal, start, width)
    noise_mean = noise_window(signal, noise_regions_)
    bounds = slice(*window_bounds(noise_regions_))
    last_noise = noise_mean
    signal[bounds, 1] -= last_noise
    while i < n:
        i += 1
        noise_mean = noise_window(signal, noise_regions(signal, start, width))
        signal[bounds, 1] -= noise_mean
        signal[bounds, 1].clip(min=0, out=signal[bounds, 1])
        if (last_noise - noise_mean) < 1e-6:
            break
        else:
            last_noise = noise_mean
    if noise_mean == 0:
        noise_mean = 1e-6
    return NoiseRegion(bounds.start, bounds.stop, noise_mean, signal)


def index_of_boundaries(signal, width=0.5, window_size=10):
    base = signal[0, 0]
    windows = [0]
    size = signal.shape[0]
    start = 0
    end = 0
    offset = 0

    window_size /= width
    window_size += 1
    window_size = int(window_size)

    while end < size:
        i = 0
        regions = []
        while i < window_size:
            lo = base + (width * (offset + i)) + -width
            hi = base + (width * (offset + i)) + width
            mask = (signal[:, 0] >= lo) & (signal[:, 0] <= hi)
            regions.append(np.where(mask)[0])
            i += 1
        offset += i
        region_sizes = np.array(map(np.shape, regions))
        # Don't accept regions of size 0
        region_sizes[region_sizes == 0] = 100000000
        try:
            last_region = regions[np.argmin(region_sizes)]
        except ValueError, e:
            print(e)
            break
        try:
            end = last_region[-1]
            windows.append(offset)
        except IndexError:
            break
    return windows[:-1]
