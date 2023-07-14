import pickle

from ms_peak_picker.peak_picker import pick_peaks, find_signal_to_noise

def test_peak_picker():
    with open('tests/data/arrays.pkl', 'rb') as fh:
        data = pickle.load(fh)

    mz_array = data['mz']
    intensity_array = data['intensity']

    assert find_signal_to_noise(
        intensity_array[15194], intensity_array, 15194) > 0

    assert find_signal_to_noise(
        intensity_array[15197], intensity_array, 15197) == 0

    peaks = pick_peaks(mz_array, intensity_array)
    assert len(peaks) == 2107
    assert abs(sum(p.intensity for p in peaks) - 4531125399.828125) < 1e-3
