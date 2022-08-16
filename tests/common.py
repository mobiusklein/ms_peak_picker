from ms_peak_picker.peak_set import FittedPeak


def make_peak(mz, intensity, fwhm=0.05):
    return FittedPeak(mz, intensity, 0, 0, 0, fwhm, intensity)
