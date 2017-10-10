# cython: embedsignature=True

cimport cython
from cython cimport parallel
cimport numpy as np
from libc cimport math
from libc.stdlib cimport malloc, calloc, free
import numpy as np

from cpython cimport PyFloat_AsDouble
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM

np.import_array()


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
        converted[i].mz = &mz[0]
        converted[i].intensity = &inten[0]
        converted[i].size = mz.shape[0]
    out[0] = converted
    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef average_signal(object arrays, double dx=0.01, object weights=None):
    cdef:
        double lo, hi
        double mz_j, mz_j1
        double inten_j, inten_j1
        double x, contrib
        double scan_weight
        size_t i, j, n, n_arrays, n_points
        long k_array, n_scans
        object omz, ointen
        list convert

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

    if weights is None:
        weights = [1 for omz in arrays]
    elif len(arrays) != len(weights):
        raise ValueError("`arrays` and `weights` must have the same length")

    convert = []
    for omz, ointen in arrays:
        mz = omz.astype(np.double)
        inten = ointen.astype(np.double)
        convert.append((mz, inten))

    lo = convert[0][0][0]
    hi = 0
    for mz, inten in convert:
        lo = min(mz[0], lo)
        hi = max(mz[mz.shape[0] - 1], hi)

    lo = max(lo - 1, 0)
    hi += 1

    n_scans = len(convert)

    pweights = <double*>malloc(sizeof(double) * n_scans)
    for i in range(n_scans):
        pweights[i] = PyFloat_AsDouble(float(weights[i]))

    prepare_arrays(convert, &spectrum_pairs)

    mz_array = np.arange(lo, hi, dx)
    intensity_array = np.zeros_like(mz_array)
    n_points = mz_array.shape[0]
    with nogil:
        for k_array in parallel.prange(n_scans):
            intensity_array_local = <double*>calloc(sizeof(double), n_points)
            pair = spectrum_pairs[k_array]
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
            with gil:
                for i in range(n_points):
                    intensity_array[i] += intensity_array_local[i]
            free(intensity_array_local)
    free(spectrum_pairs)
    free(pweights)
    return mz_array, intensity_array / sum(weights)
