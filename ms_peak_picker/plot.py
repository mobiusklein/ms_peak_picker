'''A collection of tools for drawing and annotating mass spectra
'''
from collections import namedtuple

from matplotlib import pyplot as plt
import numpy as np


from .base import PeakLike


point = namedtuple('point', ('mz', 'intensity'))


def draw_raw(mz_array, intensity_array=None, ax=None, normalize=False, **kwargs):
    """Draws un-centroided profile data, visualizing continuous
    data points

    Parameters
    ----------
    mz_array : :class:`np.ndarray` or :class:`tuple`
        Either the m/z array to be visualized, or if `intensity_array`
        is `None`, `mz_array` will be unpacked, looking to find a sequence
        of two `np.ndarray` objects for the m/z (X) and intensity (Y)
        coordinates
    intensity_array : :class:`np.ndarray`, optional
        The intensity array to be visualized. If `None`, will attempt to
        unpack `mz_array`
    ax : :class:`matplotlib.Axes`, optional
        The axis to draw the plot on. If missing, a new one will be created using
        :func:`matplotlib.pyplot.subplots`
    pretty: bool, optional
        If `True`, will call :func:`_beautify_axes` on `ax`
    normalize: bool, optional
        if `True`, will normalize the abundance dimension to be between 0 and 100%
    **kwargs
        Passed to :meth:`matplotlib.Axes.plot`

    Returns
    -------
    :class:`~.Axes`
    """
    pretty = kwargs.pop("pretty", True)
    if intensity_array is None and len(mz_array) == 2:
        mz_array, intensity_array = mz_array
    if ax is None:
        _, ax = plt.subplots(1)
    if normalize:
        intensity_array = intensity_array / intensity_array.max() * 100.0
    ax.plot(mz_array, intensity_array, **kwargs)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity")
    if pretty:
        if intensity_array.shape[0] > 0:
            set_ylim = intensity_array.min() >= 0
        else:
            set_ylim = True
        _beautify_axes(ax, set_ylim)
    return ax


def peaklist_to_vector(peaklist, width=0.000001):
    """Convert a list of discrete centroided peaks into a pair of continuous m/z
    and intensity arrays

    Parameters
    ----------
    peaklist : :class:`~Iterable` of :class:`~.PeakLike`
        The collection of peaks to convert
    width : float, optional
        The spacing between the center of the peak and it's shoulders

    Returns
    -------
    np.ndarray:
        The generated m/z array
    np.ndarray:
        The generated intensity array

    Raises
    ------
    TypeError
        When the input could not be coerced into a peak list
    """
    try:
        mzs = []
        intensities = []
        for peak in sorted(peaklist, key=lambda x: x.mz):
            mzs.append(peak.mz - width)
            intensities.append(0.)
            mzs.append(peak.mz)
            intensities.append(peak.intensity)
            mzs.append(peak.mz + width)
            intensities.append(0.)
        return np.array(mzs), np.array(intensities)
    except AttributeError:
        pt = peaklist[0]
        if not PeakLike.is_a(pt):
            try:
                if len(pt) == 2:
                    peaklist = [point(*p) for p in peaklist]
                    return peaklist_to_vector(peaklist, width)
            except Exception:
                pass
        raise TypeError("Expected a sequence of peak-like objects"
                        " or (mz, intensity) pairs, but got %r instead" % type(pt))


def draw_peaklist(peaklist, ax=None, normalize=False, **kwargs):
    """Draws centroided peak data, visualizing peak apexes.

    The peaks will be converted into a single smooth curve using
    :func:`peaklist_to_vector`.

    Parameters
    ----------
    peaklist: :class:`Iterable` of :class:`~.PeakLike`
        The peaks to draw.
    ax : matplotlib.Axes, optional
        The axis to draw the plot on. If missing, a new one will be created using
        :func:`matplotlib.pyplot.subplots`
    pretty: bool, optional
        If `True`, will call :func:`_beautify_axes` on `ax`
    normalize: bool, optional
        if `True`, will normalize the abundance dimension to be between 0 and 100%
    **kwargs
        Passed to :meth:`matplotlib.Axes.plot`

    Returns
    -------
    matplotlib.Axes
    """
    pretty = kwargs.pop("pretty", True)
    if ax is None:
        _, ax = plt.subplots(1)
    mz_array, intensity_array = peaklist_to_vector(peaklist)
    if normalize:
        intensity_array = intensity_array / intensity_array.max() * 100.0
    ax.plot(mz_array, intensity_array, **kwargs)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity")
    if pretty:
        if intensity_array.shape[0] > 0:
            set_ylim = intensity_array.min() >= 0
        else:
            set_ylim = True
        _beautify_axes(ax, set_ylim)
    return ax


def _beautify_axes(ax, set_ylim=True):
    ax.axes.spines['right'].set_visible(False)
    ax.axes.spines['top'].set_visible(False)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks_position('none')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    if set_ylim:
        ax.set_ylim(0, max(ax.get_ylim()))
    return ax


__all__ = [
    "draw_peaklist", "draw_raw"
]
