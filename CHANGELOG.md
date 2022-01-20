# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][Keep a Changelog] and this project adheres to [Semantic Versioning][Semantic Versioning].

## [Unreleased]


### Added
1. Added the `peak_statistics.zero_pad` function to fill sparse arrays with delimiting zero values, and a
   `scan_filter.ZeroFiller` (label: `zero_fill`) filter.
2. Added `ms_peak_picker.scan_averaging.GridAverager` which makes averaging a fixed set of spectra efficiently

### Changed
1. `ms_peak_picker.pick_peaks` no longer returns `None` when the input arrays are empty, instead returning
   an empty `PeakIndex`.
2. The `linear_resampling` filter now uses a 0.005 m/z spacing.
3. `ms_peak_picker.pick_peaks` will now enforce `signal_to_noise_threshold` when the input data is already centroided
   but the definition of "signal to noise" ratio differs from profile mode spectra.

### Deprecated

### Removed

### Fixed
1. `ms_peak_picker.reprofile` now does not segfault when the peak list(s) are empty.


### Security


---

## [Released]

---

<!-- Links -->
[Keep a Changelog]: https://keepachangelog.com/
[Semantic Versioning]: https://semver.org/

<!-- Versions -->
[Unreleased]: https://github.com/mobiusklein/ms_peak_picker/compare/v0.1.28...HEAD
[Released]: https://github.com/mobiusklein/ms_peak_picker/releases
[0.1.28]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.28