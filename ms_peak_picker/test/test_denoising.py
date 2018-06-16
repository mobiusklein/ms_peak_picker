import unittest

import numpy as np

from ms_peak_picker.fticr_denoising import denoise
from ms_peak_picker.test.common import make_peak
from ms_peak_picker import reprofile, pick_peaks


class TestDenoising(unittest.TestCase):
    def make_data(self):
        peak = make_peak(200, 1e4)
        x = np.arange(0, 1000, 0.01)
        y = (np.random.random(x.size) + 1) * 100
        y[19700:20300] += reprofile([peak])[1]
        return x, y

    def test_denoise(self):
        x, y = self.make_data()
        denoised_x, denoised_y = denoise(x, y, window_size=2.0)
        assert y.mean() >= 100.0
        assert denoised_y.mean() < 1.0
        assert len(pick_peaks(denoised_x, denoised_y)) == 1

    def test_no_change(self):
        x, y = self.make_data()
        denoised_x, denoised_y = denoise(x, y, window_size=2.0, scale=0)
        assert np.allclose(x, denoised_x)
        assert np.allclose(y, denoised_y)


if __name__ == '__main__':
    unittest.main()
