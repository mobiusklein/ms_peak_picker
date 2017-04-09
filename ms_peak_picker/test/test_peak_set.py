import unittest

import numpy as np

from ms_peak_picker.peak_set import FittedPeak, PeakSet, _FittedPeak, _PeakSet
from ms_peak_picker.peak_index import PeakIndex, _PeakIndex
from ms_peak_picker.peak_statistics import gaussian_shape


def make_peak(mz, intensity, fwhm=0.05):
    return FittedPeak(mz, intensity, 0, 0, 0, fwhm, intensity)


def make_profile_array(points, fwhm=0.05):
    peaks = []
    for point in points:
        fp = make_peak(*point, fwhm=fwhm)
        peaks.append(fp)
    mz = np.array([0])
    intensity = np.array([0])

    for p in peaks:
        x, y = gaussian_shape(p)
        mz = np.concatenate([mz, [x[0] - 0.0001], x, [x[-1] + 0.0001]])
        intensity = np.concatenate([intensity, [0], y, [0]])
    return mz, intensity


points = [(276.5, 2e4), (576.5, 8e4), (862.1, 15e4)]


class TestPeakSet(unittest.TestCase):
    def make_profile(self):
        return make_profile_array(points)

    @staticmethod
    def make_peaks():
        inst = PeakSet([make_peak(*point) for point in points])
        inst.reindex()
        return inst

    def test_construct(self):
        inst = self.make_peaks()
        self.assertEqual(len(inst), 3)

    def test_has_peak(self):
        inst = self.make_peaks()
        self.assertIsNotNone(inst.has_peak(576.5))
        self.assertIsNone(inst.has_peak(1576.5))

    def test_between(self):
        inst = self.make_peaks()
        self.assertIn(inst.has_peak(576.5), list(inst.between(300, 1000)))
        self.assertNotIn(inst.has_peak(276.5), list(inst.between(300, 1000)))

    def test_clone(self):
        inst = self.make_peaks()
        dup = inst.clone()
        for a, b in zip(inst.peaks, dup.peaks):
            self.assertEqual(a, b)


class TestPeakIndex(TestPeakSet):
    @staticmethod
    def make_peaks():
        inst = PeakSet([make_peak(*point) for point in points])
        inst.reindex()
        inst = PeakIndex(np.array([]), np.array([]), inst)
        return inst


if __name__ == '__main__':
    unittest.main()
