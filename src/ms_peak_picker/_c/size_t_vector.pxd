


cdef struct size_t_vector:
    size_t* v
    size_t used
    size_t size

cdef size_t_vector* make_size_t_vector_with_size(size_t size) noexcept nogil
cdef size_t_vector* make_size_t_vector() noexcept nogil
cdef int size_t_vector_resize(size_t_vector* vec) noexcept nogil
cdef int size_t_vector_append(size_t_vector* vec, size_t value) noexcept nogil
cdef int size_t_vector_reserve(size_t_vector* vec, size_t new_size) noexcept nogil
cdef void size_t_vector_reset(size_t_vector* vec) noexcept nogil

cdef void free_size_t_vector(size_t_vector* vec) noexcept nogil
cdef void print_size_t_vector(size_t_vector* vec) noexcept nogil

cdef class SizeTVector(object):
    cdef __cythonbufferdefaults__ = {'ndim' : 1, 'mode':'c'}

    cdef:
        size_t_vector* impl
        int flags

    cdef int allocate_storage(self) noexcept nogil
    cdef int allocate_storage_with_size(self, size_t size) noexcept nogil

    cdef int free_storage(self) noexcept nogil
    cdef bint get_should_free(self) noexcept nogil
    cdef void set_should_free(self, bint flag) noexcept nogil

    cdef size_t* get_data(self) noexcept nogil

    @staticmethod
    cdef SizeTVector _create(size_t size)

    @staticmethod
    cdef SizeTVector wrap(size_t_vector* vector)

    cdef size_t get(self, size_t i) noexcept nogil
    cdef void set(self, size_t i, size_t value) noexcept nogil
    cdef size_t size(self) noexcept nogil
    cdef int cappend(self, size_t value) noexcept nogil

    cdef SizeTVector _slice(self, object slice_spec)

    cpdef SizeTVector copy(self)

    cpdef int append(self, object value) except *
    cpdef int extend(self, object values) except *

    cpdef int reserve(self, size_t size) noexcept nogil

    cpdef int fill(self, size_t value) noexcept nogil


    cpdef void qsort(self, bint reverse=?) noexcept nogil

    cpdef object _to_python(self, size_t value)
    cpdef size_t _to_c(self, object value) except *