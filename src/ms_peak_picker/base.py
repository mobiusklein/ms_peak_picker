import six
import abc


def simple_repr(self):  # pragma: no cover
    template = "{self.__class__.__name__}({d})"
    d = [
        "%s=%r" % (k, v) if v is not self else "(...)" for k, v in sorted(
            self.__dict__.items(), key=lambda x: x[0])
        if (not k.startswith("_") and not callable(v)) and not (k == "signal")]
    return template.format(self=self, d=', '.join(d))


class Base(object):
    __repr__ = simple_repr


@six.add_metaclass(abc.ABCMeta)
class PeakLike(object):

    @classmethod
    def __subclasshook__(cls, C):
        if cls is PeakLike:
            mz = any(tuple(getattr(B, 'mz', None) is not None for B in C.mro()))
            intensity = any(tuple(getattr(B, 'intensity', None) is not None for B in C.mro()))
            return mz and intensity
        return NotImplemented

    @classmethod
    def is_a(cls, obj):
        val = isinstance(obj, cls)
        if val:
            return val
        val = hasattr(obj, 'mz') and hasattr(obj, 'intensity')
        return val
