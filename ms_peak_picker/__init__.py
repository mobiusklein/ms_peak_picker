import os
from .peak_picker import PeakProcessor, pick_peaks, fit_type_map, peak_mode_map
from .peak_set import PeakSet, FittedPeak, is_peak, simple_peak
from .peak_index import PeakIndex
from . import peak_statistics
from . import search
from . import fticr_denoising
from . import fft_patterson_charge_state
from . import scan_filter
from .base import PeakLike
from .reprofile import reprofile
from .scan_averaging import average_signal
from .smoothing import gaussian_smooth
from .version import version as __version__

__all__ = ["PeakProcessor", "pick_peaks", "PeakIndex", "PeakSet",
           "FittedPeak", "peak_statistics", "search", "fticr_denoising",
           "scan_filter", "fft_patterson_charge_state", "get_include",
           "fit_type_map", "peak_mode_map", "reprofile", "average_signal",
           "gaussian_smooth", 'is_peak', 'simple_peak', "PeakLike"]


def get_include():
    """Retrieve the path to compiled C extensions' source files to make linking simple.
    """
    return os.path.join(__path__[0], "_c")


try:
    from ms_peak_picker._c import peak_picker as _cpeak_picker
    has_c = True
    _has_c_error = None
except ImportError as _has_c_error:
    has_c = False


def check_c_extensions():
    if has_c:
        print("C extensions appear to have imported successfully.")
        return True
    else:
        print("Could not import peak picking machinery: %r" % (_has_c_error, ))
    return False
