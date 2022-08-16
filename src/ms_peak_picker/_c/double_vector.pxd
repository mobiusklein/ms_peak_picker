cdef struct DoubleVector:
    double* v
    size_t used
    size_t size

cdef DoubleVector* make_double_vector_with_size(size_t size) nogil
cdef DoubleVector* make_double_vector() nogil
cdef int double_vector_resize(DoubleVector* vec) nogil
cdef int double_vector_append(DoubleVector* vec, double value) nogil
cdef void free_double_vector(DoubleVector* vec) nogil
cdef void print_double_vector(DoubleVector* vec) nogil
cdef void reset_double_vector(DoubleVector* vec) nogil
cdef list double_vector_to_list(DoubleVector* vec)
cdef DoubleVector* list_to_double_vector(list input_list)