cimport numpy as np

ctypedef np.float64_t DTYPE_t

cpdef size_t nearest_left(np.ndarray[DTYPE_t, ndim=1] vec, DTYPE_t target_val, size_t start_index=*)
cpdef size_t nearest_right(np.ndarray[DTYPE_t, ndim=1] vec, DTYPE_t target_val, size_t start_index=*)
cpdef size_t get_nearest_binary(np.ndarray[DTYPE_t, ndim=1] vec, DTYPE_t target_val, size_t start_index=*, object stop_index=*)
cpdef size_t get_nearest(np.ndarray[DTYPE_t, ndim=1] vec, DTYPE_t target_val, size_t start_index)
