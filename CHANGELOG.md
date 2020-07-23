# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][Keep a Changelog] and this project adheres to [Semantic Versioning][Semantic Versioning].

## [Unreleased]

### Added

### Changed
1. `ms_peak_picker.pick_peaks` no longer returns `None` when the input arrays are empty, instead returning
   an empty `PeakIndex`.
2. The `linear_resampling` filter now uses a 0.005 m/z spacing.

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