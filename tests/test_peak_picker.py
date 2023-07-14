import pickle

from ms_peak_picker.peak_picker import pick_peaks

def test_peak_picker():
    with open('tests/data/arrays.pkl', 'rb') as fh:
        data = pickle.load(fh)

    mz = data['mz']
    intensity = data['intensity']
    peaks = pick_peaks(mz, intensity)
    assert len(peaks) == 2108
    assert abs(sum(p.intensity for p in peaks) - 4531158140.658203) < 1e-3
