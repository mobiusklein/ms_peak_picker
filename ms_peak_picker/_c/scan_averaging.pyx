# cython: embedsignature=True

cimport cython
from cython cimport parallel
cimport numpy as np
from libc cimport math
from libc.stdlib cimport malloc, calloc, free
from multiprocessing import cpu_count
import numpy as np

from cpython cimport PyFloat_AsDouble
from cpython cimport PyInt_AsLong
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from cpython.slice cimport PySlice_GetIndicesEx

from ms_peak_picker._c.double_vector cimport (
    make_double_vector_with_size, make_double_vector,
    double_vector_append, reset_double_vector,
    free_double_vector, DoubleVector)

np.import_array()

cdef long num_processors = PyInt_AsLong(cpu_count())


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


cdef struct spectrum_holder:
    double* mz
    double* intensity
    size_t size


cdef int prepare_arrays(list arrays, spectrum_holder** out):
    cdef:
        size_t i, n
        spectrum_holder* converted
        np.ndarray[double, ndim=1, mode='c'] mz
        np.ndarray[double, ndim=1, mode='c'] inten

    n = len(arrays)
    converted = <spectrum_holder*>malloc(sizeof(spectrum_holder) * n)
    if converted == NULL:
        return 1
    for i in range(n):
        mz, inten = arrays[i]
        converted[i].size = mz.shape[0]
        if mz.shape[0]:
            converted[i].mz = &mz[0]
            converted[i].intensity = &inten[0]
        else:
            converted[i].mz = NULL
            converted[i].intensity = NULL
    out[0] = converted
    return 0


cdef double INF = float('inf')


cdef class GridAverager(object):
    cdef:
        public np.ndarray mz_axis
        public double min_mz
        public double max_mz
        public double dx
        double* mz_axis_
        size_t size
        size_t width
        public np.ndarray intensities
        public np.ndarray empty


    def __init__(self, min_mz, max_mz, width, dx=0.001, arrays=None):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.dx = dx
        self.width = width
        self.init_mz_axis()
        self.init_intensities_grid()

    cdef void init_mz_axis(self):
        cdef:
            np.ndarray[double, ndim=1, mode='c'] mz_axis
        self.mz_axis = mz_axis = np.arange(self.min_mz, self.max_mz + self.dx, self.dx)
        self.size = mz_axis.shape[0]
        self.mz_axis_ = &mz_axis[0]

    cdef void init_intensities_grid(self):
        self.intensities = np.zeros((self.width, self.size), dtype=np.float64)
        self.empty = np.zeros(self.width, dtype=np.uint8)

    cpdef add_spectrum(self, np.ndarray[double, ndim=1, mode='c'] mz_array, np.ndarray[double, ndim=1, mode='c'] intensity_array, size_t index):
        rebinned_intensities = self.create_intensity_axis(mz_array, intensity_array, index)
        self.empty[index] = mz_array.shape[0] == 0
        return rebinned_intensities

    @cython.cdivision(True)
    cdef int _populate_intensity_axis(self, double* pmz, double* pinten, double* intensity_axis, size_t n, size_t m) nogil:
        cdef:
            double x, mz_j, mz_j1, contrib, inten_j, inten_j1
            size_t i, j

        if m == 0:
            return 0

        for i in range(n):
            x = self.mz_axis_[i]
            j = binsearch(&pmz[0], x, m)
            mz_j = pmz[j]
            if mz_j < x and j + 1 < m:
                mz_j1 = pmz[j + 1]
                inten_j = pinten[j]
                inten_j1 = pinten[j + 1]
            elif mz_j > x and j > 0:
                mz_j1 = mz_j
                inten_j1 = pinten[j]
                mz_j = pmz[j - 1]
                inten_j = pmz[j - 1]
            else:
                continue

            contrib = ((inten_j * (mz_j1 - x)) + (inten_j1 * (x - mz_j))) / (mz_j1 - mz_j)
            intensity_axis[i] += contrib
        return 0

    @cython.cdivision(True)
    cpdef np.ndarray create_intensity_axis(self, np.ndarray[double, ndim=1, mode='c'] mz_array, np.ndarray[double, ndim=1, mode='c'] intensity_array, size_t index):
        cdef:
            double* pmz
            double* pinten
            np.ndarray[double, ndim=1, mode='c'] intensity_axis
            double* intensity_axis_
            double x, mz_j, mz_j1, contrib, inten_j, inten_j1
            size_t i, j, n, m
            double[:, :] intensity_grid

        intensity_grid = self.intensities
        intensity_axis = intensity_grid[index, :]
        intensity_axis_ = &intensity_axis[0]

        m = mz_array.shape[0]
        if m == 0:
            return intensity_axis

        pmz = &mz_array[0]
        pinten = &intensity_array[0]
        n = self.size

        with nogil:
            self._populate_intensity_axis(pmz, pinten, intensity_axis_, n, m)
        return intensity_axis

    @cython.boundscheck(False)
    cpdef add_spectra(self, list spectra, n_workers=None):
        cdef:
            spectrum_holder* spectrum_pairs
            spectrum_holder pair
            double[:, :] intensity_grid
            np.uint8_t[:] emptiness
            double[:] intensity_frame
            int n_threads, i, n

        if n_workers is None:
            n_workers = min(num_processors, len(spectra))
        n_threads = PyInt_AsLong(n_workers)


        prepare_arrays(spectra, &spectrum_pairs)
        n = PyList_GET_SIZE(spectra)

        intensity_grid = self.intensities
        emptiness = self.empty

        with nogil:
            for i in parallel.prange(n, num_threads=n_threads):
                pair = spectrum_pairs[i]
                emptiness[i] = pair.size == 0
                if emptiness[i]:
                    continue
                self._populate_intensity_axis(pair.mz, pair.intensity, &intensity_grid[i, 0], self.size, pair.size)

            free(spectrum_pairs)


    @cython.cdivision(True)
    cpdef np.ndarray[double, ndim=1, mode='c'] average_indices(self, slice slc):
        cdef:
            Py_ssize_t start, stop, step, slicelength, n
            size_t i, j
            np.ndarray[double, ndim=1, mode='c'] intensity_axis
            np.ndarray[double, ndim=1, mode='c'] intensity_frame
            double* intensity_frame_
            double* intensity_axis_
            bint is_empty
            double normalizer
            np.uint8_t[:] emptiness
            double[:, :] intensity_grid

        n = PyList_GET_SIZE(self.intensities)
        if PySlice_GetIndicesEx(slc, n, &start, &stop, &step, &slicelength) == -1:
            raise ValueError("Invalid slice")

        intensity_axis = np.zeros_like(self.mz_axis)
        intensity_axis_ = &intensity_axis[0]
        normalizer = stop - start
        if normalizer == 0:
            normalizer = 1

        emptiness = self.empty
        intensity_grid = self.intensities

        for i in range(<size_t>start, <size_t>stop):
            is_empty = emptiness[i]
            if is_empty:
                continue
            for j in range(self.size):
                intensity_axis_[j] += intensity_grid[i, j] / normalizer
        return intensity_axis



@cython.cdivision(True)
@cython.boundscheck(False)
cpdef average_signal(object arrays, double dx=0.01, object weights=None, object num_threads=None):
    """Average multiple spectras' intensity arrays, with a common m/z axis

    Parameters
    ----------
    arrays : :class:`list` of pairs of :class:`np.ndarray`
        The m/z and intensity arrays to combine
    dx : float, optional
        The m/z resolution to build the averaged m/z axis with
    weights : :class:`list` of :class:`float`, optional
        Weight of each entry in `arrays`. Defaults to 1.0 for each if not provided.

    Returns
    -------
    mz_array: :class:`np.ndarray`
    intensity_array: :class:`np.ndarray`
    """
    cdef:
        double lo, hi
        double mz_j, mz_j1
        double inten_j, inten_j1
        double x, contrib
        double scan_weight
        size_t i, j, n, n_arrays, n_points
        long k_array, n_scans, n_workers, worker_block
        object omz, ointen
        list convert
        int n_empty
        bint all_empty

        spectrum_holder* spectrum_pairs
        spectrum_holder pair

        np.ndarray[double, ndim=1, mode='c'] mz_array
        np.ndarray[double, ndim=1, mode='c'] intensity_array
        np.ndarray[double, ndim=1, mode='c'] mz
        np.ndarray[double, ndim=1, mode='c'] inten
        double* pweights
        double* pmz
        double* pinten
        double* intensity_array_local
        double* pintensity_array_total
        double** intensity_layers
    if num_threads is None or num_threads < 0:
        n_workers = num_processors
    else:
        n_workers = num_threads
    if weights is None:
        weights = [1 for omz in arrays]
    elif len(arrays) != len(weights):
        raise ValueError("`arrays` and `weights` must have the same length")

    convert = []
    for omz, ointen in arrays:
        if len(omz) == 0:
            continue
        mz = omz.astype(np.double)
        inten = ointen.astype(np.double)
        convert.append((mz, inten))

    n_scans = len(convert)
    if n_scans == 0:
        return np.arange(0., 0.), np.arange(0., 0.)
    lo = INF
    hi = 0
    for mz, inten in convert:
        lo = min(mz[0], lo)
        hi = max(mz[mz.shape[0] - 1], hi)

    lo = max(lo - 1, 0)
    hi += 1

    pweights = <double*>malloc(sizeof(double) * n_scans)
    for i in range(n_scans):
        pweights[i] = PyFloat_AsDouble(float(weights[i]))

    prepare_arrays(convert, &spectrum_pairs)

    mz_array = np.arange(lo, hi, dx)
    intensity_array = np.zeros_like(mz_array)
    n_points = mz_array.shape[0]
    pintensity_array_total = &(intensity_array[0])
    with nogil:
        if n_scans < n_workers:
            n_workers = n_scans
        intensity_layers = <double**>malloc(sizeof(double*) * n_scans)
        for k_array in parallel.prange(n_scans, num_threads=n_workers):
            intensity_layers[k_array] = intensity_array_local = <double*>calloc(sizeof(double), n_points)
            pair = spectrum_pairs[k_array]
            if pair.size == 0:
                continue
            pmz = pair.mz
            pinten = pair.intensity
            contrib = 0
            scan_weight = pweights[k_array]
            for i in range(n_points):
                x = mz_array[i]
                j = binsearch(pmz, x, pair.size)
                mz_j = pmz[j]
                if mz_j < x and j + 1 < pair.size:
                    mz_j1 = pmz[j + 1]
                    inten_j = pinten[j]
                    inten_j1 = pinten[j + 1]
                elif mz_j > x and j > 0:
                    mz_j1 = mz_j
                    inten_j1 = pinten[j]
                    mz_j = pmz[j - 1]
                    inten_j = pmz[j - 1]
                else:
                    continue

                contrib = ((inten_j * (mz_j1 - x)) + (inten_j1 * (x - mz_j))) / (mz_j1 - mz_j)
                intensity_array_local[i] += contrib * scan_weight

        for k_array in range(n_scans):
            intensity_array_local = intensity_layers[k_array]
            for i in range(n_points):
                pintensity_array_total[i] += intensity_array_local[i]
            free(intensity_array_local)
        free(intensity_layers)
        scan_weight = 0
        for i in range(n_scans):
            scan_weight += pweights[i]
        for i in range(n_points):
            pintensity_array_total[i] /= scan_weight
        free(spectrum_pairs)
        free(pweights)
    return mz_array, intensity_array
