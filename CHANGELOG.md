# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][Keep a Changelog] and this project adheres to [Semantic Versioning][Semantic Versioning].

## [Unreleased]

### Added
-

### Changed

### Deprecated

### Removed

### Fixed

### Security


## [v0.1.35] - 2022-02-15

### Added

### Changed
- Registered Python versions supported changed to Py37-39

### Deprecated

### Removed

### Fixed

### Security


## [v0.1.34] - 2022-02-14


### Added
1. Added the `peak_statistics.zero_pad` function to fill sparse arrays with delimiting zero values, and a
   `scan_filter.ZeroFiller` (label: `zero_fill`) filter.
2. Added `ms_peak_picker.scan_averaging.GridAverager` which makes averaging a fixed set of spectra more efficient

### Changed
1. `ms_peak_picker.pick_peaks` no longer returns `None` when the input arrays are empty, instead returning
   an empty `PeakIndex`.
2. The `linear_resampling` filter now uses a 0.005 m/z spacing.
3. `ms_peak_picker.pick_peaks` will now enforce `signal_to_noise_threshold` when the input data is already centroided
   but the definition of "signal to noise" ratio differs from profile mode spectra.
4. `average_signal` now accepts a parameter `num_threads` to control the number of threads launched with OpenMP. Odd numbers
   work best.

### Deprecated

### Removed

### Fixed
1. `ms_peak_picker.reprofile` now does not segfault when the peak list(s) are empty.
2. Fixed segfault when `ms_peak_picker._c.peak_statistics.curve_reg_dv` encounters a singular matrix


### Security


---

## [Released]

---

<!-- Links -->
[Keep a Changelog]: https://keepachangelog.com/
[Semantic Versioning]: https://semver.org/

<!-- Versions -->
[Unreleased]: https://github.com/mobiusklein/ms_peak_picker/compare/v0.1.34...HEAD
[Released]: https://github.com/mobiusklein/ms_peak_picker/releases
[0.1.28]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.28
[0.1.34]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.34