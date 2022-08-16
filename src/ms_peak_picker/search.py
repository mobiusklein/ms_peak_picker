import numpy as np


def nearest_left(vec, target_val, start_index=0):
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
    nearest_index = start_index
    next_index = start_index

    if next_index == 0:
        return 0
    next_val = vec[next_index]
    best_distance = np.fabs(target_val - next_val)
    while next_val > target_val:
        next_index -= 1
        next_val = vec[next_index]
        dist = np.fabs(next_val - target_val)
        if dist < best_distance:
            best_distance = dist
            nearest_index = next_index
        if next_index == 0:
            break
    return nearest_index


def nearest_right(vec, target_val, start_index=0):
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
    nearest_index = start_index
    next_index = start_index

    size = len(vec) - 1
    if next_index == size:
        return size

    next_val = vec[next_index]
    best_distance = np.abs(next_val - target_val)
    while (next_val < target_val):
        next_index += 1
        next_val = vec[next_index]
        dist = np.fabs(next_val - target_val)
        if dist < best_distance:
            best_distance = dist
            nearest_index = next_index
        if next_index == size:
            break
    return nearest_index


def get_nearest_binary(vec, target_val, start_index=0, stop_index=None):
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
    if stop_index is None:
        stop_index = len(vec) - 1
    if vec[start_index] > target_val:
        return start_index
    if vec[stop_index] < target_val:
        return stop_index
    while True:
        min_val = vec[start_index]
        max_val = vec[stop_index]
        if np.fabs(stop_index - start_index) <= 1 and (target_val >= min_val) and (target_val <= max_val):
            if np.fabs(min_val - target_val) < np.fabs(max_val - target_val):
                return start_index
            return stop_index
        ratio = (max_val - target_val) / (max_val - min_val)
        mid_index = int(start_index * ratio + stop_index * (1 - ratio) + 0.5)
        if (mid_index == start_index):
            mid_index = start_index + 1
        elif (mid_index == stop_index):
            mid_index = stop_index - 1

        mid_val = vec[mid_index]

        if mid_val >= target_val:
            stop_index = mid_index
        elif mid_index + 1 == stop_index:
            if np.fabs(mid_val - target_val) < np.fabs(max_val - target_val):
                return mid_index
            return stop_index
        else:
            mid_next_val = vec[mid_index + 1]
            if target_val >= mid_val and target_val <= mid_next_val:
                if (target_val - mid_val < mid_next_val - mid_val):
                    return mid_index
                return mid_index + 1
            start_index = mid_index + 1


def get_nearest(vec, target_val, start_index):
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
    size = len(vec) - 1
    if target_val >= vec[size]:
        return size
    elif target_val <= vec[0]:
        return 0

    distance_remaining = target_val - vec[start_index]

    if start_index < size:
        step = vec[start_index + 1] - vec[start_index]
    else:
        step = vec[start_index] - vec[start_index - 1]
    step = float(step)

    move_by = int(distance_remaining / step)

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


_has_c = True

try:

    _nearest_left = nearest_left
    _nearest_right = nearest_right
    _get_nearest_binary = get_nearest_binary
    _get_nearest = get_nearest
    from ms_peak_picker._c.search import (
        nearest_left, nearest_right, get_nearest_binary, get_nearest)
except ImportError:
    pass
