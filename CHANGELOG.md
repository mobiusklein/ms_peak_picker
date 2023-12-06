# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][Keep a Changelog] and this project adheres to [Semantic Versioning][Semantic Versioning].

## [v0.1.44] - 2023-12-05

### Fixed
1. Fixed Cython code compatibility with Python 3.11. See [cython/issues/5894](https://github.com/cython/cython/issues/5894)
   and [cpython/issues/112768](https://github.com/python/cpython/issues/112768)

## [v0.1.43] - 2023-10-12

### Fixed
1. Fixed problem with conversion of a pre-centroided list of peaks with repeated m/z values near the end of
   the mass range, or where the peak list is a single peak long.


## [v0.1.42] - 2023-08-04

### Fixed
1. Fixed Cython 3 compatibility. There is now the potential for errors that were previously undetected
   to error out with a traceback.


### Changed
1. The `plot` module will more aggressively use scientific notation offsets to simplify y tick labels.


## [v0.1.41] - 2023-07-14

### Fixed
1. Fixed a non-determinism problem during sorting of the m/z and intensity arrays prior to picking peaks.

## [v0.1.40] - 2022-10-09

### Changed
1. The required NumPy version for Python 3.10+ is now 1.23.2


## [v0.1.38] - 2022-03-24

### Added

### Changed

### Deprecated

### Removed

### Fixed
- Properly skip adjacent m/z points at equal intensity to the most recently fit peak.

### Security


## [v0.1.37] - 2022-03-21

### Added

### Changed

### Deprecated

### Removed

### Fixed
- `PeakSet.between` properly behaves for query m/z values beyond either end of the `PeakSet`'s m/z range

### Security


## [v0.1.36] - 2022-03-21

### Added
- Added fast path for skipping search for m/z index coordinates when picking peaks.
- Added method to `get_occupied_intervals` for `GridAverager` to get coordinates on the m/z axis.

### Changed

### Deprecated

### Removed

### Fixed
- `PeakSet.all_peaks_for` properly behaves for query m/z values beyond either end of the `PeakSet`'s m/z range

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
[Unreleased]: https://github.com/mobiusklein/ms_peak_picker/compare/v0.1.44...HEAD
[Released]: https://github.com/mobiusklein/ms_peak_picker/releases
[0.1.28]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.28
[v0.1.34]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.34
[v0.1.35]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.35
[v0.1.36]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.36
[v0.1.37]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.37
[v0.1.38]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.38
[v0.1.39]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.39
[v0.1.40]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.40
[v0.1.41]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.41
[v0.1.42]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.42
[v0.1.43]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.43
[v0.1.44]: https://github.com/mobiusklein/ms_peak_picker/releases/v0.1.44