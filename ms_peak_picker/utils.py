try:
    range = xrange
except:
    range = range

from collections import defaultdict
import numpy as np


def simple_repr(self):  # pragma: no cover
    template = "{self.__class__.__name__}({d})"
    d = [
        "%s=%r" % (k, v) if v is not self else "(...)" for k, v in sorted(
            self.__dict__.items(), key=lambda x: x[0])
        if (not k.startswith("_") and not callable(v)) and not (k == "signal")]
    return template.format(self=self, d=', '.join(d))


class Base(object):
    __repr__ = simple_repr


def ppm_error(x, y):
    return (x - y) / y


def gaussian_volume(peak):
    center = peak.mz
    amplitude = peak.intensity
    fwhm = peak.full_width_at_half_max
    spread = fwhm / 2.35482
    x = np.arange(center - fwhm - 0.02, center + fwhm + 0.02, 0.001)
    return x, amplitude * np.exp(-((x - center) ** 2) / (2 * spread ** 2))


def add_peak_volume(a, axis=None):
    xa, ya = gaussian_volume(a)
    if axis is None:
        axis = defaultdict(float)
    for x, y in zip(xa, ya):
        axis[x] += y
    return axis


def peaklist_to_profile(peaks, precision=5):
    axis = defaultdict(float)
    for p in peaks:
        add_peak_volume(p, axis)
    axis_xs = defaultdict(float)
    for x, y in axis.items():
        axis_xs[round(x, precision)] += y
    xs, ys = map(np.array, zip(*sorted(axis_xs.items())))
    return xs, ys

try:
    has_plot = True
    from matplotlib import pyplot as plt

    def draw_raw(mz_array, intensity_array=None, ax=None, **kwargs):
        if intensity_array is None and len(mz_array) == 2:
            mz_array, intensity_array = mz_array
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(mz_array, intensity_array, **kwargs)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Intensity")
        return ax

    def peaklist_to_vector(peaklist):
        mzs = []
        intensities = []
        for peak in sorted(peaklist, key=lambda x: x.mz):
            mzs.append(peak.mz - .000001)
            intensities.append(0.)
            mzs.append(peak.mz)
            intensities.append(peak.intensity)
            mzs.append(peak.mz + .000001)
            intensities.append(0.)
        return np.array(mzs), np.array(intensities)

    def draw_peaklist(peaklist, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        mz_array, intensity_array = peaklist_to_vector(peaklist)
        ax.plot(mz_array, intensity_array, **kwargs)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Intensity")
        return ax

except ImportError:
    has_plot = False
