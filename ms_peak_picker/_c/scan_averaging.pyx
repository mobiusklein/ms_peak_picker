# cython: embedsignature=True

cimport cython
from cython cimport parallel
cimport numpy as cnp
from libc cimport math
from libc.stdlib cimport malloc, calloc, free
from multiprocessing import cpu_count
import numpy as np

cdef extern from * nogil:
    int printf (const char *template, ...)

from cpython cimport PyFloat_AsDouble, PyInt_AsLong, PyErr_SetString, PyErr_Format
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from cpython.slice cimport PySlice_GetIndicesEx

from ms_peak_picker._c.double_vector cimport (
    make_double_vector_with_size, make_double_vector,
    double_vector_append, reset_double_vector,
    free_double_vector, DoubleVector)


from ms_peak_picker._c.size_t_vector cimport (
    make_size_t_vector_with_size, make_size_t_vector,
    size_t_vector_append, free_size_t_vector,
    size_t_vector, size_t_vector_reset, SizeTVector
)


from ms_peak_picker._c.interval_t_vector cimport (
    interval_t, initialize_interval_vector_t,
    make_interval_t_vector_with_size, make_interval_t_vector,
    interval_t_vector_append, free_interval_t_vector,
    interval_t_vector, interval_t_vector_reset, IntervalVector
)


cnp.import_array()

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


cdef int prepare_arrays(list arrays, spectrum_holder** out) except 1:
    cdef:
        size_t i, n
        spectrum_holder* converted
        cnp.ndarray[double, ndim=1, mode='c'] mz
        cnp.ndarray[double, ndim=1, mode='c'] inten

    n = len(arrays)
    converted = <spectrum_holder*>malloc(sizeof(spectrum_holder) * n)
    if converted == NULL:
        PyErr_Format(MemoryError, "Failed to allocate array holder of size %lu", n)
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
        public cnp.ndarray mz_axis
        public double min_mz
        public double max_mz
        public double dx
        double* mz_axis_
        public size_t size
        public size_t num_scans
        public cnp.ndarray intensities
        public cnp.ndarray empty
        interval_t_vector* occupied_indices


    def __init__(self, min_mz, max_mz, num_scans, dx=0.001, arrays=None):
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.dx = dx
        self.num_scans = num_scans
        self.occupied_indices = NULL
        self.init_mz_axis()
        self.init_intensities_grid()

    cdef void init_mz_axis(self):
        cdef:
            cnp.ndarray[double, ndim=1, mode='c'] mz_axis
        self.mz_axis = mz_axis = np.arange(self.min_mz, self.max_mz + self.dx, self.dx)
        self.size = mz_axis.shape[0]
        self.mz_axis_ = &mz_axis[0]

    cdef int init_intensities_grid(self) except 1:
        cdef:
            size_t i
            int code

        self.intensities = np.zeros((self.num_scans, self.size), dtype=np.float64)
        self.empty = np.zeros(self.num_scans, dtype=np.uint8)
        self.occupied_indices = <interval_t_vector*>malloc(sizeof(interval_t_vector) * self.num_scans)
        if self.occupied_indices == NULL:
            PyErr_Format(MemoryError, "Failed to allocate index array of size %lu", self.num_scans)
            return 1
        for i in range(self.num_scans):
            code = initialize_interval_vector_t(&self.occupied_indices[i], 2)
            if code == 1:
                PyErr_Format(MemoryError, "Failed to allocate index array segment at %lu", i)
                return 1
        return 0

    cpdef release(self):
        cdef:
            size_t i

        if self.occupied_indices != NULL:
            for i in range(self.num_scans):
                free(self.occupied_indices[i].v)
            free(self.occupied_indices)
            self.occupied_indices = NULL


    def __dealloc__(self):
        self.release()

    cpdef IntervalVector get_occupied_indices(self, size_t i):
        if i > self.num_scans:
            raise IndexError(i)
        return IntervalVector.wrap(&self.occupied_indices[i])

    cpdef add_spectrum(self, cnp.ndarray[double, ndim=1, mode='c'] mz_array, cnp.ndarray[double, ndim=1, mode='c'] intensity_array, size_t index):
        self.empty[index] = mz_array.shape[0] == 0
        rebinned_intensities = self.create_intensity_axis(mz_array, intensity_array, index)
        return rebinned_intensities

    @cython.cdivision(True)
    cdef int _populate_intensity_axis(self, double* pmz, double* pinten, double* intensity_axis, size_t n, size_t m, interval_t_vector* occupied_acc) nogil:
        cdef:
            double x, mz_j, mz_j1, contrib, inten_j, inten_j1
            size_t i, j
            bint opened
            interval_t current_interval

        if m == 0:
            return 0
        current_interval.start = 0
        current_interval.end = 0
        opened = False
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
            if intensity_axis[i] > 0:
                if opened:
                    current_interval.end = i
                else:
                    current_interval.start = i
                    opened = True
            else:
                if opened:
                    current_interval.end = i
                    opened = False
                    if interval_t_vector_append(occupied_acc, current_interval) == -1:
                        return -1
        if opened:
            current_interval.end = i
            opened = False
            if interval_t_vector_append(occupied_acc, current_interval) == -1:
                return -1
        return 0

    @cython.cdivision(True)
    @cython.boundscheck(True)
    cpdef cnp.ndarray create_intensity_axis(self, cnp.ndarray[double, ndim=1, mode='c'] mz_array, cnp.ndarray[double, ndim=1, mode='c'] intensity_array, size_t index):
        cdef:
            double* pmz
            double* pinten
            cnp.ndarray[double, ndim=1, mode='c'] intensity_axis
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
            interval_t_vector_reset(&self.occupied_indices[index])
            if self._populate_intensity_axis(pmz, pinten, intensity_axis_, n, m, &self.occupied_indices[index]) == -1:
                with gil:
                    raise ValueError("Failed to populate intensity axis")
        return intensity_axis

    @cython.boundscheck(True)
    cpdef add_spectra(self, list spectra, n_workers=None):
        cdef:
            spectrum_holder* spectrum_pairs
            spectrum_holder pair
            double[:, ::1] intensity_grid
            cnp.uint8_t[::1] emptiness
            double[:] intensity_frame
            int n_threads, i, n
            size_t start, stop

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
                interval_t_vector_reset(&self.occupied_indices[i])
                if emptiness[i]:
                    continue
                if self._populate_intensity_axis(
                        pair.mz, pair.intensity, &intensity_grid[i, 0], self.size, pair.size, &self.occupied_indices[i]) == -1:
                    with gil:
                        raise ValueError("Failed to populate intensity axis")

            free(spectrum_pairs)


    @cython.cdivision(True)
    @cython.boundscheck(True)
    @cython.wraparound(False)
    cpdef cnp.ndarray[double, ndim=1, mode='c'] average_indices(self, size_t start, size_t stop, n_workers=None):
        cdef:
            size_t i, n, k
            interval_t_vector occupied
            interval_t current_interval
            cnp.ndarray[double, ndim=1, mode='c'] intensity_axis
            cnp.ndarray[double, ndim=1, mode='c'] intensity_frame
            double* intensity_frame_
            # double* intensity_axis_
            double[::1] intensity_axis_
            bint is_empty
            double normalizer
            cnp.uint8_t[::1] emptiness
            double[:, ::1] intensity_grid
            int n_threads
            ssize_t z
            long j, n_j

        if n_workers is None:
            n_threads = 1
        else:
            n_threads = PyInt_AsLong(n_workers)

        if start > stop:
            stop, start = start, stop
        if stop > self.num_scans:
            stop = self.num_scans

        n = PyList_GET_SIZE(self.intensities)

        intensity_axis = np.zeros_like(self.mz_axis)
        # intensity_axis_ = &intensity_axis[0]
        intensity_axis_ = intensity_axis
        normalizer = stop - start
        if normalizer == 0:
            normalizer = 1

        emptiness = self.empty
        intensity_grid = self.intensities

        z = self.size

        with nogil:
            for i in range(start, stop):
                is_empty = emptiness[i]
                if is_empty:
                    continue
                occupied = self.occupied_indices[i]
                n_j = occupied.used
                for j in parallel.prange(n_j, num_threads=n_threads):
                    current_interval = occupied.v[j]
                    for k in range(current_interval.start, current_interval.end):
                        if i >= intensity_grid.shape[0]:
                            printf("Overrun axis 0 %lld/%lld\n", i, intensity_grid.shape[0])
                        if j >= intensity_grid.shape[1]:
                            printf("Overrun axis 0 %lld/%lld\n", k, intensity_grid.shape[1])
                        intensity_axis_[k] += intensity_grid[i, k] / normalizer
        return intensity_axis

    def __getitem__(self, i):
        if isinstance(i, slice):
            start = i.start or 0
            stop = i.stop or self.num_scans
        else:
            start = i - 1
            stop = i + 2
        if start < 0:
            start = 0
        if stop > self.num_scans:
            stop = self.num_scans
        return self.mz_axis, self.average_indices(start, stop)




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
    num_threads : int, optional
        The maximum number of threads to use while averaging signal. Defaults
        to the number of spectra being averaged or the maximum available from
        the hardware, whichever is smaller.

    Returns
    -------
    mz_array: :class:`np.ndarray`
    intensity_array: :class:`np.ndarray`
    """
    cdef:
        double lo, hi
        double mz_j, mz_j1
        double inten_j, inten_j1
        double x, contrib, tmp
        double scan_weight
        size_t i, j, n, n_arrays, n_points
        long k_array, n_scans, n_workers, worker_block
        object omz, ointen
        list convert
        int n_empty
        bint all_empty
        int error

        spectrum_holder* spectrum_pairs
        spectrum_holder pair

        cnp.ndarray[double, ndim=1, mode='c'] mz_array
        cnp.ndarray[double, ndim=1, mode='c'] intensity_array
        cnp.ndarray[double, ndim=1, mode='c'] mz
        cnp.ndarray[double, ndim=1, mode='c'] inten
        double* pweights
        double* pmz
        double* pinten
        double* intensity_array_local
        double[::1] pmz_array
        double[::1] pintensity_array_total
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

    n_scans = PyList_GET_SIZE(convert)
    if n_scans == 0:
        return np.array([], dtype=np.double), np.array([], dtype=np.double)
    lo = INF
    hi = 0
    for mz, inten in convert:
        lo = min(mz[0], lo)
        hi = max(mz[mz.shape[0] - 1], hi)

    lo = max(lo - 1, 0)
    hi += 1
    pweights = <double*>malloc(sizeof(double) * n_scans)
    if pweights == NULL:
        printf("Unable to allocate memory for average_signal, could not allocate weights array of size %ld\n", n_scans)
        raise MemoryError("Unable to allocate memory for average_signal, could not allocate weights array of size %d" % (n_scans, ))
    for i in range(n_scans):
        pweights[i] = PyFloat_AsDouble(float(weights[i]))

    mz_array = np.arange(lo, hi, dx, dtype=np.double)
    intensity_array = np.zeros_like(mz_array, dtype=np.double)
    n_points = len(mz_array)
    pintensity_array_total = intensity_array
    pmz_array = mz_array
    error = 0
    intensity_layers = <double**>malloc(sizeof(double*) * n_scans)
    if intensity_layers == NULL:
        free(pweights)
        printf("Unable to allocate memory for average_signal, could not allocate temporary array of size %ld\n", n_scans)
        raise MemoryError("Unable to allocate memory for average_signal, could not allocate temporary array of size %d" % (n_scans, ))

    for k_array in range(n_scans):
        intensity_layers[k_array] = NULL

    for k_array in range(n_scans):
        intensity_layers[k_array] = <double*>malloc(sizeof(double) * n_points)
        if intensity_layers[k_array] == NULL:
            printf("Unable to allocate temporary array %d of size %zu for average_signal\n", k_array, n_points)
            error += 1

    if error:
        for k_array in range(n_scans):
            if intensity_layers[k_array] != NULL:
                free(intensity_layers[k_array])
        free(pweights)
        free(intensity_layers)
        printf("Unable to allocate memory for average_signal, failed %d partitions of size %zu\n", error, n_points)
        raise MemoryError("Unable to allocate memory for average_signal, failed %d partitions of size %d" % (error, n_points))

    prepare_arrays(convert, &spectrum_pairs)
    with nogil:
        if n_scans < n_workers:
            n_workers = n_scans
        for k_array in parallel.prange(n_scans, num_threads=n_workers):
            intensity_array_local = intensity_layers[k_array]
            pair = spectrum_pairs[k_array]
            if pair.size == 0:
                continue
            pmz = pair.mz
            pinten = pair.intensity
            contrib = 0
            scan_weight = pweights[k_array]
            for i in range(n_points):
                x = pmz_array[i]
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
                tmp = mz_j1 - mz_j
                if tmp == 0:
                    contrib = 0.0
                else:
                    contrib = ((inten_j * (mz_j1 - x)) + (inten_j1 * (x - mz_j))) / (mz_j1 - mz_j)
                intensity_array_local[i] = contrib * scan_weight

        for k_array in range(n_scans):
            intensity_array_local = intensity_layers[k_array]
            if intensity_array_local == NULL:
                continue
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
    if error:
        raise MemoryError("Unable to allocate memory for average_signal, failed %d partitions of size %d" % (error, n_points))
    return mz_array, intensity_array
