import unittest

import numpy as np

from ms_peak_picker import reprofile
from ms_peak_picker.test.common import make_peak
from ms_peak_picker import pick_peaks


class TestReprofile(unittest.TestCase):
    def make_data(self):
        peak = make_peak(200, 1e4)
        return [peak]

    def test_reprofile(self):
        peaks = self.make_data()
        x, y = reprofile(peaks)
        repicked = pick_peaks(x, y)
        diff = repicked[0].mz - peaks[0].mz
        assert abs(diff) < 1e-3

    def test_empty(self):
        peaks = []
        x, y = reprofile(peaks)
        repicked = pick_peaks(x, y)
        assert len(repicked) == 0

        peaks = self.make_data()
        peaks[0].full_width_at_half_max = 10
        x, y = reprofile(peaks, max_fwhm=1.0)
        repicked = pick_peaks(x, y)
        assert len(repicked) == 0
