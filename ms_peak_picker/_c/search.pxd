cimport cython
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.float64_t mz_t

cpdef size_t nearest_left(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=*)
cpdef size_t nearest_right(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=*)
cpdef size_t get_nearest_binary(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index=*, object stop_index=*)
cpdef size_t get_nearest(np.ndarray[mz_t, ndim=1, mode='c'] vec, DTYPE_t target_val, size_t start_index)
