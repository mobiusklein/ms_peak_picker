from libc.stdlib cimport malloc, realloc, free
from libc cimport *

cdef extern from * nogil:
    int printf (const char *template, ...)


cdef DoubleVector* make_double_vector_with_size(size_t size) nogil:
    cdef:
        DoubleVector* vec

    vec = <DoubleVector*>malloc(sizeof(DoubleVector))
    vec.v = <double*>malloc(sizeof(double) * size)
    vec.size = size
    vec.used = 0

    return vec


cdef DoubleVector* make_double_vector() nogil:
    return make_double_vector_with_size(4)


cdef int double_vector_resize(DoubleVector* vec) nogil:
    cdef:
        size_t new_size
        double* v
    new_size = vec.size * 2
    v = <double*>realloc(vec.v, sizeof(double) * new_size)
    if v == NULL:
        printf("double_vector_resize returned -1\n")
        return -1
    vec.v = v
    vec.size = new_size
    return 0


cdef int double_vector_append(DoubleVector* vec, double value) nogil:
    if (vec.used + 1) == vec.size:
        double_vector_resize(vec)
    vec.v[vec.used] = value
    vec.used += 1
    return 0


cdef void free_double_vector(DoubleVector* vec) nogil:
    free(vec.v)
    free(vec)


cdef void print_double_vector(DoubleVector* vec) nogil:
    cdef:
        size_t i
    i = 0
    printf("[")
    while i < vec.used:
        printf("%0.6f", vec.v[i])
        if i != (vec.used - 1):
            printf(", ")
        i += 1
    printf("]\n")


cdef void reset_double_vector(DoubleVector* vec) nogil:
    vec.used = 0


cdef list double_vector_to_list(DoubleVector* vec):
    cdef:
        size_t i
        list result
    result = list()
    i = 0
    while i < vec.used:
        result.append(vec.v[i])
        i += 1
    return result


cdef DoubleVector* list_to_double_vector(list input_list):
    cdef:
        DoubleVector* vec
        double val
    vec = make_double_vector_with_size(len(input_list))
    for val in input_list:
        double_vector_append(vec, val)
    return vec


def test():
    cdef list listy
    cdef DoubleVector* vecty

    listy = [1., 2., 3., 4., 23121.]
    vecty = list_to_double_vector(listy)
    print_double_vector(vecty)
    listy = []
    print(listy)
    listy = double_vector_to_list(vecty)
    free_double_vector(vecty)
    print(listy)
    del listy
