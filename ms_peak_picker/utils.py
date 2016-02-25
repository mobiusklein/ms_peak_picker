try:
    range = xrange
except:
    range = range


def simple_repr(self):  # pragma: no cover
    template = "{self.__class__.__name__}({d})"
    d = [
        "%s=%r" % (k, v) if v is not self else "(...)" for k, v in sorted(
            self.__dict__.items(), key=lambda x: x[0])
        if (not k.startswith("_") and not callable(v)) and not (k == "signal")]
    return template.format(self=self, d=', '.join(d))


class Base(object):
    __repr__ = simple_repr


def ppm_error(x, y):
    return (x - y) / y

try:
    has_plot = True
    from matplotlib import pyplot as plt

    def draw_raw(mz_array, intensity_array, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(mz_array, intensity_array, **kwargs)
        ax.xaxis.set_ticks_position('none')
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Intensity")
        return ax

    def draw_peaklist(peaklist, ax=None, **kwargs):
        kwargs.setdefault("width", 0.01)
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.bar([p.mz for p in peaklist], [p.intensity for p in peaklist], **kwargs)
        ax.xaxis.set_ticks_position('none')
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Intensity")
        return ax

except ImportError:
    has_plot = False
