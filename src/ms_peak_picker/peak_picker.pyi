
class PartialPeakFitState:
    full_width_at_half_max: float
    left_width: float
    right_width: float
    set: bool
    signal_to_noise: float

    def reset(self) -> None: ...


class PeakProcessor:
    background_intensity: float
    fit_type: str
    intensity_threshold: float
    partial_fit_state: PartialPeakFitState
    peak_data: list
    peak_mode: str
    signal_to_noise_threshold: float
    threshold_data: bool
    verbose: bool
