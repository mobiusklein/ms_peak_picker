import unittest

import numpy as np

from ms_peak_picker.peak_set import FittedPeak, PeakSet, _has_c, _PeakSet
from ms_peak_picker.peak_index import PeakIndex
from ms_peak_picker.reprofile import reprofile
from ms_peak_picker.peak_picker import pick_peaks, PeakProcessor, _has_c as _has_peak_picker_c

from ms_peak_picker.test.common import make_peak


points = [(276.5, 2e4), (576.5, 8e4), (862.1, 15e4)]


class TestPeakSet(unittest.TestCase):
    peak_set_type = PeakSet

    def make_peaks(self):
        inst = self.peak_set_type([make_peak(*point) for point in points])
        inst.reindex()
        return inst

    def test_construct(self):
        inst = self.make_peaks()
        self.assertEqual(len(inst), 3)

    def test_has_peak(self):
        inst = self.make_peaks()
        self.assertIsNotNone(inst.has_peak(576.5))
        self.assertIsNone(inst.has_peak(1576.5))
        for peak in inst:
            assert inst.has_peak(peak.mz) is not None

    def test_between(self):
        inst = self.make_peaks()
        self.assertIn(inst.has_peak(576.5), list(inst.between(300.0, 1000.01)))
        self.assertNotIn(inst.has_peak(276.5), list(inst.between(300.0, 1000.01)))

    def test_clone(self):
        inst = self.make_peaks()
        dup = inst.clone()
        for a, b in zip(inst.peaks, dup.peaks):
            self.assertEqual(a, b)


if _has_c:
    class TestPythonPeakSet(TestPeakSet):
        peak_set_type = _PeakSet


class TestPeakPicker(unittest.TestCase):
    @staticmethod
    def make_profile():
        return reprofile([make_peak(*point) for point in points])

    def test_peak_processor(self):
        mzs, intensities = self.make_profile()
        proc = PeakProcessor()
        proc.discover_peaks(mzs, intensities)
        peaks = proc.peak_data
        assert len(peaks) == 3

    def test_python_impl_peak_processor(self):
        if _has_peak_picker_c:
            from ms_peak_picker.peak_picker import _PeakProcessor
            mzs, intensities = self.make_profile()
            proc = _PeakProcessor()
            proc.discover_peaks(mzs, intensities)
            peaks = proc.peak_data
            assert len(peaks) == 3

    def test_pick_peaks(self):
        mzs, intensities = self.make_profile()
        peaks = pick_peaks(mzs, intensities)
        peak = peaks.has_peak(276.5, 1e-5)
        self.assertIsNotNone(peak)


class TestPeakIndex(TestPeakSet):
    @staticmethod
    def make_peaks():
        inst = PeakSet([make_peak(*point) for point in points])
        inst.reindex()
        inst = PeakIndex(np.array([]), np.array([]), inst)
        return inst


if __name__ == '__main__':
    unittest.main()
