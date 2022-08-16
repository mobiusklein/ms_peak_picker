try:
    range = xrange
except NameError:
    range = range


from .base import Base


def ppm_error(x, y):
    return (x - y) / y


try:
    has_plot = True
    from .plot import (draw_peaklist, draw_raw)
except (RuntimeError, ImportError):
    has_plot = False


__all__ = [
    "Base", "ppm_error", "has_plot",
    "draw_raw", "draw_peaklist"
]
