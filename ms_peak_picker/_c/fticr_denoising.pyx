# cython: embedsignature=True

cimport cython
from cython cimport parallel
cimport numpy as np
from libc cimport math
from libc.math cimport fabs, sqrt, log, ceil, floor, log
from libc.stdlib cimport malloc, calloc, free
import numpy as np

from cpython cimport PyFloat_AsDouble
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM

np.import_array()

ctypedef double mz_type
ctypedef double intensity_type
ctypedef np.uint64_t count_type


@cython.cdivision(True)
@cython.boundscheck(False)
cdef double log2(double x) nogil:
    return log(x) / log(2)


@cython.cdivision(True)
@cython.boundscheck(False)
cdef size_t binsearch(double* array, double x, size_t n) nogil:
    cdef:
        size_t lo, hi, mid
        double y, err

    lo = 0
    hi = n

    while hi != lo:
        mid = (hi + lo) / 2
        y = array[mid]
        err = y - x
        if hi - lo == 1:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
cdef double percentile(intensity_type[:] N, double percent) nogil:
    cdef:
        double k, f, c, d0, d1
    k = (N.shape[0] - 1) * percent
    f = floor(k)
    c = ceil(k)
    if f == c:
        return N[<size_t>(k)]
    d0 = N[<size_t>(f)] * (c - k)
    d1 = N[<size_t>(c)] * (k - f)
    return d0 + d1


@cython.cdivision(True)
@cython.boundscheck(False)
cdef double freedman_diaconis_bin_width(intensity_type[:] x) nogil:
    cdef:
        double q75, q25, iqr

    q75 = percentile(x, 0.75)
    q25 = percentile(x, 0.25)
    iqr = q75 - q25
    return 2.0 * iqr * (x.shape[0] ** (-1.0 / 3.0))


@cython.cdivision(True)
@cython.boundscheck(False)
cdef double sturges_bin_width(intensity_type[:] x) nogil:
    cdef:
        double d
        double mn, mx
    d = log2(x.shape[0] + 1.0)
    minmax(x, &mn, &mx)
    return (mx - mn) / d


@cython.cdivision(True)
@cython.boundscheck(False)
cdef int minmax(intensity_type[:] x, intensity_type* minimum, intensity_type* maximum) nogil:
    cdef:
        size_t i, n
        intensity_type d, lo, hi

    n = x.shape[0]
    if n == 0:
        minimum[0] = 0
        maximum[0] = 0
        return 0
    lo = x[0]
    hi = x[0]

    for i in range(x.shape[0]):
        d = x[i]
        if d < lo:
            lo = d
        if d > hi:
            hi = d
    minimum[0] = lo
    maximum[0] = hi
    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
cdef int _histogram(intensity_type[:] a, count_type* bin_count, intensity_type* bin_edges, int bins=10) nogil:
    cdef:
        size_t i, n, j
        intensity_type mn, mx, norm, x
        int hit
        intensity_type binwidth, binwidth_fd, binwidth_st

    minmax(a, &mn, &mx)

    if mn == mx:
        mn -= 0.5
        mx += 0.5

    binwidth = (mx - mn) / bins

    norm = bins / (mx - mn)

    for i in range(bins + 1):
        bin_edges[i] = i * binwidth

    n = a.shape[0]
    for i in range(n):
        x = a[i]
        hit = 0
        for j in range(1, bins + 1):
            binwidth = bin_edges[j]
            if x < binwidth:
                hit = 1
                bin_count[j - 1] += 1
                break
        if not hit:
            bin_count[j - 1] += 1
    return 0


def histogram(intensity_type[:] a, int bins=10):
    cdef:
        np.ndarray[count_type, ndim=1, mode='c'] bin_count
        np.ndarray[intensity_type, ndim=1, mode='c'] bin_edges
    bin_count = np.zeros(bins, dtype=np.uint64)
    bin_edges = np.empty(bins + 1)
    _histogram(a, &bin_count[0], &bin_edges[0], bins)
    return bin_count, bin_edges


cdef class Window(object):
    cdef:
        public mz_type[:] mz_array
        public intensity_type[:] intensity_array
        public ssize_t start_index
        public ssize_t end_index
        public mz_type center_mz
        public intensity_type mean_intensity
        public ssize_t size
        mz_type* _mz_array
        intensity_type* _intensity_array

        int bins
        count_type* _bin_count
        intensity_type* _bin_edges

    def __init__(self, np.ndarray[mz_type, ndim=1, mode='c'] mz_array,
                 np.ndarray[intensity_type, ndim=1, mode='c'] intensity_array,
                 start_index, end_index, center_mz, bins=10):
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.start_index = start_index
        self.end_index = end_index
        self.center_mz = center_mz
        self._mz_array = &mz_array[0]
        self._intensity_array = &intensity_array[0]
        self.size = mz_array.shape[0]
        self.mean_intensity = -1
        self.bins = bins
        self._init_arrays()
        self.deduct_intensity(0)

    def __dealloc__(self):
        self._release_arrays()

    cdef void _init_arrays(self):
        self._release_arrays()
        self._bin_count = <count_type*>calloc(sizeof(count_type), self.bins)
        self._bin_edges = <intensity_type*>malloc(sizeof(intensity_type) * (self.bins + 1))

    cdef void _release_arrays(self):
        if self._bin_edges != NULL:
            free(self._bin_edges)
            self._bin_edges = NULL
        if self._bin_count != NULL:
            free(self._bin_count)
            self._bin_count = NULL

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _deduct_intensity(self, double value):
        cdef:
            size_t i, n
            double total
        total = 0
        n = self.size
        for i in range(n):
            self._intensity_array[i] -= value
            if self._intensity_array[i] < 0:
                self._intensity_array[i] = 0
            total += self._intensity_array[i]
        if n == 0:
            n = 1
        self.mean_intensity = total / n

    cpdef deduct_intensity(self, double value):
        self._deduct_intensity(value)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef double truncated_mean(self, double threshold=0.95):
        cdef:
            double mask_level, total, weight
            size_t i_count, n_count
        if self.size == 0:
            return 1e-6
        _histogram(self.intensity_array, self._bin_count, self._bin_edges, self.bins)

        n_count = self.bins
        mask_level = 0
        for i_count in range(n_count):
            if self._bin_count[i_count] > mask_level:
                mask_level = self._bin_count[i_count]
        mask_level *= (1 - threshold)
        total = 0
        weight = 0
        for i_count in range(n_count):
            if mask_level < self._bin_count[i_count]:
                total += self._bin_edges[i_count + 1] * self._bin_count[i_count]
                weight += self._bin_count[i_count]
        return total / weight

    def __repr__(self):
        return "Window(start_index=%d, end_index=%d, mean_intensity=%f, center_mz=%f)" % (
            self.start_index, self.end_index, self.mean_intensity, self.center_mz)


cdef class NoiseRegion(object):
    cdef:
        public list windows
        public double width
        public size_t start_index
        public size_t end_index
        public size_t size


    def __init__(self, windows, width=10):
        self.windows = windows
        self.width = width
        self.start_index = windows[0].start_index
        self.end_index = windows[-1].end_index
        self.size = PyList_GET_SIZE(windows)

    def __repr__(self):
        return "NoiseRegion(start_index=%d, end_index=%d)" % (
            self.start_index, self.end_index)

    cpdef Window noise_window(self):
        cdef:
            size_t i, n
            double minimum
            Window minimum_window, window

        i = 0
        n = self.size
        if n == 0:
            return None
        minimum_window = self.getitem(0)
        minimum = minimum_window.mean_intensity
        for i in range(1, n):
            window = self.getitem(i)
            if window.mean_intensity < minimum:
                minimum_window = window
                minimum = minimum_window.mean_intensity
        return minimum_window

    def arrays(self):
        mz_array = np.concatenate([w.mz_array for w in self.windows])
        intensity_array = np.concatenate(
            [w.intensity_array for w in self.windows])
        return mz_array, intensity_array

    @cython.final
    cdef Window getitem(self, size_t i):
        return <Window>PyList_GET_ITEM(self.windows, i)

    cpdef double denoise(self, double scale=5, int maxiter=10):
        cdef:
            double noise_mean, last_mean
            size_t i, n, niter

        if scale == 0:
            return 0
        noise_mean = self.noise_window().truncated_mean() * scale
        n = self.size
        for i in range(n):
            window = self.getitem(i)
            window.deduct_intensity(noise_mean)
        last_mean = noise_mean
        noise_mean = self.noise_window().truncated_mean()
        niter = 1
        while abs(last_mean - noise_mean) > 1e-3 or niter < maxiter:
            niter += 1
            noise_mean = self.noise_window().truncated_mean() * scale
            for i in range(n):
                window = self.getitem(i)
                window._deduct_intensity(noise_mean)
            last_mean = noise_mean

        return last_mean - noise_mean


cpdef list windowed_spectrum(np.ndarray[mz_type, ndim=1, mode='c'] mz_array,
                             np.ndarray[intensity_type, ndim=1, mode='c'] intensity_array,
                             double window_size=1.):
    cdef:
        mz_type mz_min, mz_max, step_size, center_mz
        mz_type lo_mz, hi_mz
        ssize_t center_i, niter, lo_i, hi_i, i, n
        list windows 
        Window win
        mz_type* _mz_array

    n = mz_array.shape[0]
    _mz_array = &mz_array[0]
    if n < 2:
        return []
    mz_min = _mz_array[0]
    mz_max = _mz_array[n - 1]


    step_size = window_size / 2.
    center_mz = mz_min + step_size
    center_i = binsearch(_mz_array, center_mz, n)

    windows = []

    niter = 0

    while center_mz < mz_max:
        lo_mz = center_mz - step_size
        hi_mz = center_mz + step_size

        lo_i = binsearch(_mz_array, lo_mz, n)
        hi_i = binsearch(_mz_array, hi_mz, n)
        win = Window(mz_array[lo_i:hi_i + 1],
                     intensity_array[lo_i:hi_i + 1],
                     lo_i, hi_i, center_mz)
        windows.append(win)

        center_mz = center_mz + window_size
        center_i = binsearch(_mz_array, center_mz, n)

        niter += 1

    return windows


cpdef list group_windows(list windows, int width=10):
    cdef:
        int step
        list regions
        size_t i, n

    step = int(width / 2)
    regions = []
    i = step
    n = len(windows)
    while i < n:
        lo = i - step
        hi = i + step
        reg = NoiseRegion(windows[lo:hi])
        regions.append(reg)
        i += step * 2
    return regions


cdef class FTICRScan(object):
    cdef:
        public np.ndarray mz_array
        public np.ndarray intensity_array
        public list windows
        public list regions

    def __init__(self, mz_array, intensity_array):
        self.mz_array = mz_array.astype(float)
        self.intensity_array = intensity_array.astype(float)

        self.windows = None
        self.regions = None

    def __iter__(self):
        yield self.mz_array
        yield self.intensity_array

    def denoise(self, window_size=1., region_width=10, scale=5):
        cdef:
            size_t i, n
            NoiseRegion region
            list mz, intensity
            object mz_, intensity_
        self.windows = windowed_spectrum(
            self.mz_array, self.intensity_array, window_size=window_size)
        self.regions = group_windows(self.windows, region_width)
        mz = []
        intensity = []
        n = len(self.regions)
        for i in range(n):
            region = self.regions[i]
            region.denoise(scale)
            mz_, intensity_ = region.arrays()
            mz.extend(mz_)
            intensity.extend(intensity_)
        return FTICRScan(np.array(mz), np.array(intensity))
