import ms_peak_picker


def test_c_extensions_load():
    assert ms_peak_picker.check_c_extensions()
