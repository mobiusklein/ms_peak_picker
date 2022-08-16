


cdef struct size_t_vector:
    size_t* v
    size_t used
    size_t size

cdef size_t_vector* make_size_t_vector_with_size(size_t size) nogil
cdef size_t_vector* make_size_t_vector() nogil
cdef int size_t_vector_resize(size_t_vector* vec) nogil
cdef int size_t_vector_append(size_t_vector* vec, size_t value) nogil
cdef int size_t_vector_reserve(size_t_vector* vec, size_t new_size) nogil
cdef void size_t_vector_reset(size_t_vector* vec) nogil

cdef void free_size_t_vector(size_t_vector* vec) nogil
cdef void print_size_t_vector(size_t_vector* vec) nogil

cdef class SizeTVector(object):
    cdef __cythonbufferdefaults__ = {'ndim' : 1, 'mode':'c'}

    cdef:
        size_t_vector* impl
        int flags

    cdef int allocate_storage(self) nogil
    cdef int allocate_storage_with_size(self, size_t size) nogil

    cdef int free_storage(self) nogil
    cdef bint get_should_free(self) nogil
    cdef void set_should_free(self, bint flag) nogil

    cdef size_t* get_data(self) nogil

    @staticmethod
    cdef SizeTVector _create(size_t size)

    @staticmethod
    cdef SizeTVector wrap(size_t_vector* vector)

    cdef size_t get(self, size_t i) nogil
    cdef void set(self, size_t i, size_t value) nogil
    cdef size_t size(self) nogil
    cdef int cappend(self, size_t value) nogil

    cdef SizeTVector _slice(self, object slice_spec)

    cpdef SizeTVector copy(self)

    cpdef int append(self, object value) except *
    cpdef int extend(self, object values) except *

    cpdef int reserve(self, size_t size) nogil

    cpdef int fill(self, size_t value) nogil


    cpdef void qsort(self, bint reverse=?) nogil

    cpdef object _to_python(self, size_t value)
    cpdef size_t _to_c(self, object value) except *