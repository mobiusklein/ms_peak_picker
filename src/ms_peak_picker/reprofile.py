import numpy as np

from ms_peak_picker import FittedPeak
from ms_peak_picker.peak_statistics import GaussianModel
from ms_peak_picker.peak_set import simple_peak, is_peak


class PeakSetReprofiler(object):

    def __init__(self, models, dx=0.01):
        self.models = sorted(models, key=lambda x: x.center)
        self.dx = dx
        self.gridx = None
        self.gridy = None
        self._build_grid()

    def _build_grid(self):
        lo = self.models[0].center
        hi = self.models[-1].center
        self.gridx = np.arange(max(lo - 3, 0), hi + 3, self.dx, dtype=np.float64)
        self.gridy = np.zeros_like(self.gridx, dtype=np.float64)

    def _reprofile(self):
        nmodels = len(self.models)
        for i, x in enumerate(self.gridx):
            y = 0
            if i % 1000 == 0:
                print(i, x)
            offset = self._find_starting_model(x)
            j = offset - 1
            while j > 0:
                model = self.models[j]
                if (x - model.center) > 3:
                    break
                pred = model.predict(x)
                y += pred
                j -= 1
            j = offset
            while j < nmodels:
                model = self.models[j]
                if (model.center - x) > 3:
                    break
                pred = model.predict(x)
                y += pred
                j += 1

            self.gridy[i] = y

    def _find_starting_model(self, x):
        lo = 0
        hi = len(self.models)
        while (lo != hi):
            mid = (hi + lo) // 2
            model = self.models[mid]
            center = model.peak.mz
            err = center - x
            if abs(err) < 0.0001:
                return mid
            elif (hi - lo) == 1:
                return mid
            elif err > 0:
                hi = mid
            else:
                lo = mid

    def reprofile(self):
        self._reprofile()
        return self.gridx, self.gridy


def models_from_peak_sets(peak_sets, max_fwhm=0.2, model_cls=None, default_fwhm=0.1, override_fwhm=None):
    if model_cls is None:
        model_cls = GaussianModel
    models = []
    for peaks in peak_sets:
        for peak in peaks:
            try:
                if peak.full_width_at_half_max > max_fwhm:
                    continue
            except AttributeError:
                peak = simple_peak(peak.mz, peak.intensity, default_fwhm)
            if override_fwhm is not None:
                peak = peak.clone()
                peak.full_width_at_half_max = override_fwhm
            models.append(model_cls(peak))
    return models


def reprofile(peaks, max_fwhm=0.2, dx=0.01, model_cls=GaussianModel, default_fwhm=0.1, override_fwhm=None):
    """Converts fitted peak centroids into theoretical profiles derived from
    its fitted parameters and a theoretical shape model.

    Parameters
    ----------
    peaks : Iterable of FittedPeak or Iterable of Iterable of FittedPeak
        The peaks to convert back into profiles. If a list of peaks is
        provided, those peaks will be converted into profiles. If a list
        of lists of peaks is provided, all peaks will be summed when
        reconstructing the total signal at a given coordinate, and an
        average will be taken.
    max_fwhm : float, optional
        The maximum full width at half max to consider when selecting
        peaks to contribute to the modeled profiles.
    dx : float, optional
        The spacing of the m/z grid to use.
    model_cls : type, optional
        A type descending from :class:`ms_peak_picker.peak_statistics.PeakShapeModel`.
        Defaults to :class:`ms_peak_picker.peak_statistics.GaussianModel`.

    Returns
    -------
    mz_array: np.ndarray[float64]
        The m/z grid reconstructed from the fitted peaks
    intensity_array: np.ndarray[float64]
        The modeled total signal at each grid point
    """
    if not peaks:
        return np.array([], dtype=float), np.array([], dtype=float)
    if is_peak(peaks[0]):
        peaks = [peaks]
    models = models_from_peak_sets(
        peaks, max_fwhm, model_cls, default_fwhm, override_fwhm)
    if not models:
        return np.array([], dtype=float), np.array([], dtype=float)
    task = PeakSetReprofiler(models, dx)
    x, y = task.reprofile()
    y /= len(peaks)
    return x, y


try:
    from ms_peak_picker._c.peak_statistics import PeakSetReprofiler
except ImportError:
    pass
