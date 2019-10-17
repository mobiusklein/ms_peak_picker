# cython: embedsignature=True

cimport numpy as np
cimport cython

from libc.math cimport fabs
# ctypedef cython.floating mz_t

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef size_t nearest_left(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=0):
    """Locate the index nearest to `target_val` in `vec` searching
    to the left of `start_index`

    Parameters
    ----------
    vec : np.array
        The array to search
    target_val : float
        The value to search for
    start_index : int, optional
        The starting point to search from

    Returns
    -------
    int
        The nearest index
    """
    cdef:
        size_t nearest_index, next_index
        DTYPE_t next_val, best_distance, dist
        mz_t* cvec
    nearest_index = start_index
    next_index = start_index

    if next_index == 0:
        return 0
    cvec = &vec[0]
    next_val = cvec[next_index]
    best_distance = fabs(target_val - next_val)
    while next_val > target_val:
        next_index -= 1
        next_val = cvec[next_index]
        dist = fabs(next_val - target_val)
        if dist < best_distance:
            best_distance = dist
            nearest_index = next_index
        if next_index == 0:
            break
    return nearest_index


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef size_t nearest_right(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=0):
    """Locate the index nearest to `target_val` in `vec` searching
    to the right of `start_index`

    Parameters
    ----------
    vec : np.array
        The array to search
    target_val : float
        The value to search for
    start_index : int, optional
        The starting point to search from

    Returns
    -------
    int
        The nearest index
    """
    cdef:
        size_t nearest_index, next_index, size
        DTYPE_t next_val, best_distance, dist
        mz_t* cvec
    nearest_index = start_index
    next_index = start_index

    size = vec.shape[0] - 1
    if next_index >= size:
        return size
    cvec = &vec[0]
    next_val = cvec[next_index]
    best_distance = fabs(next_val - target_val)
    while (next_val < target_val):
        next_index += 1
        next_val = cvec[next_index]
        dist = fabs(next_val - target_val)
        if dist < best_distance:
            best_distance = dist
            nearest_index = next_index
        if next_index == size:
            break
    return nearest_index


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef size_t get_nearest_binary(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=0, object stop_index=None):
    """Find the nearest index to `target_val` in `vec` using binary
    search

    Parameters
    ----------
    vec : np.ndarray
        The array to search
    target_val : float
        The value to search for
    start_index : int, optional
        The lower bound index
    stop_index : None, optional
        The upper bound index

    Returns
    -------
    int
        The nearest index
    """
    cdef:
        DTYPE_t min_val, max_val, ratio, mid_next_val, mid_val, cval
        size_t stop_index_, mid_index

    if stop_index is None:
        stop_index_ = vec.shape[0] - 1
    else:
        stop_index_ = stop_index
    cval = vec[start_index]
    if cval > target_val:
        return start_index

    cval = vec[stop_index_]
    if cval < target_val:
        return stop_index_
    while True:
        min_val = vec[start_index]
        max_val = vec[stop_index_]
        if fabs(stop_index_ - start_index) <= 1 and (target_val >= min_val) and (target_val <= max_val):
            if fabs(min_val - target_val) < fabs(max_val - target_val):
                return start_index
            return stop_index_
        ratio = (max_val - target_val) / (max_val - min_val)
        mid_index = <size_t>(start_index * ratio + stop_index_ * (1 - ratio) + 0.5)
        if (mid_index == start_index):
            mid_index = start_index + 1
        elif (mid_index == stop_index_):
            mid_index = stop_index_ - 1

        mid_val = vec[mid_index]

        if mid_val >= target_val:
            stop_index_ = mid_index
        elif mid_index + 1 == stop_index_:
            if fabs(mid_val - target_val) < fabs(max_val - target_val):
                return mid_index
            return stop_index_
        else:
            mid_next_val = vec[mid_index + 1]
            if target_val >= mid_val and target_val <= mid_next_val:
                if (target_val - mid_val < mid_next_val - mid_val):
                    return mid_index
                return mid_index + 1
            start_index = mid_index + 1


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef size_t get_nearest(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index):
    """Locate the index nearest to `target_val` in `vec`. Starts
    by interpolating from the value at `start_index` to `target_val`
    and then starts searching to the left and right of the guessed
    index.

    Parameters
    ----------
    vec : np.array
        The array to search
    target_val : float
        The value to search for
    start_index : int, optional
        The starting point to search from

    Returns
    -------
    int
        The nearest index
    """
    cdef:
        size_t size, next_index, move_by
        DTYPE_t step, next_val
        DTYPE_t distance_remaining


    size = (vec.shape[0]) - 1
    if target_val >= vec[size]:
        return size
    elif target_val <= vec[0]:
        return 0

    distance_remaining = target_val - vec[start_index]

    if start_index < size:
        step = vec[start_index + 1] - vec[start_index]
    else:
        step = vec[start_index] - vec[start_index - 1]

    move_by = <size_t>(distance_remaining / step)

    next_index = start_index + move_by

    if next_index < 0:
        next_index = 0
    if next_index > size:
        next_index = size - 1

    next_val = vec[next_index]
    if target_val >= next_val:
        return nearest_right(vec, target_val, next_index)
    else:
        return nearest_left(vec, target_val, next_index)
