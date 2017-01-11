import numpy as np

from .utils import Base
from .fticr_denoising import denoise as fticr_remove_baseline

filter_register = {}


def register(name, *args, **kwargs):
    def wrap(cls):
        filter_register[name] = cls(*args, **kwargs)
        return cls
    return wrap


class FilterBase(Base):

    def filter(self, mz_array, intensity_array):
        return mz_array, intensity_array

    def __call__(self, mz_array, intensity_array):
        return self.filter(mz_array, intensity_array)


@register("median")
class MedianIntensityFilter(FilterBase):

    def filter(self, mz_array, intensity_array):
        mask = intensity_array < np.median(intensity_array)
        intensity_array = np.array(intensity_array)
        intensity_array[mask] = 0.
        return mz_array, intensity_array


@register("mean_below_mean")
class MeanBelowMeanFilter(FilterBase):

    def filter(self, mz_array, intensity_array):
        mean = intensity_array.mean()
        mean_below_mean = (intensity_array < mean).mean()
        mask = intensity_array < mean_below_mean
        intensity_array[mask] = 0.
        return mz_array, intensity_array


@register("savitsky_golay")
class SavitskyGolayFilter(FilterBase):
    def __init__(self, window_length=5, polyorder=3, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def filter(self, mz_array, intensity_array):
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(
            intensity_array, window_length=self.window_length,
            polyorder=self.polyorder, deriv=self.deriv).clip(0)
        mask = smoothed > 0
        return mz_array[mask], smoothed[mask]


@register("tenth_percent_of_max")
@register("one_percent_of_max", p=0.01)
class NPercentOfMaxFilter(FilterBase):
    def __init__(self, p=0.001):
        self.p = p

    def filter(self, mz_array, intensity_array):
        mask = (intensity_array / intensity_array.max()) < self.p
        intensity_array_clone = np.array(intensity_array)
        intensity_array_clone[mask] = 0.
        return mz_array, intensity_array_clone


@register("fticr_baseline")
class FTICRBaselineRemoval(FilterBase):
    def __init__(self, window_length=1., region_width=10, scale=5):
        self.window_length = window_length
        self.region_width = region_width
        self.scale = scale

    def filter(self, mz_array, intensity_array):
        return fticr_remove_baseline(mz_array, intensity_array, self.window_length, self.region_width, self.scale)


@register("linear_resampling", 0.01)
class LinearResampling(FilterBase):
    def __init__(self, spacing):
        self.spacing = spacing

    def filter(self, mz_array, intensity_array):
        lo = mz_array.min()
        hi = mz_array.max()
        new_mz = np.arange(lo, hi + self.spacing, self.spacing)
        new_intensity = np.interp(new_mz, mz_array, intensity_array)
        return new_mz, new_intensity


@register("over_10", 10)
@register("over_100", 100)
class ConstantThreshold(FilterBase):
    def __init__(self, constant):
        self.constant = constant

    def filter(self, mz_array, intensity_array):
        mask = intensity_array > self.constant
        return mz_array[mask], intensity_array[mask]


def transform(mz_array, intensity_array, filters=None):
    if filters is None:
        filters = []

    for filt in filters:
        if isinstance(filt, basestring):
            filt = filter_register[filt]
        mz_array, intensity_array = filt(mz_array, intensity_array)

    return mz_array, intensity_array
