

cdef struct interval_t:
    size_t start
    size_t end


cdef struct interval_t_vector:
    interval_t* v
    size_t used
    size_t size

cdef int initialize_interval_vector_t(interval_t_vector* vec, size_t size) nogil

cdef interval_t_vector* make_interval_t_vector_with_size(size_t size) nogil
cdef interval_t_vector* make_interval_t_vector() nogil
cdef int interval_t_vector_resize(interval_t_vector* vec) nogil
cdef int interval_t_vector_append(interval_t_vector* vec, interval_t value) nogil
cdef int interval_t_vector_reserve(interval_t_vector* vec, size_t new_size) nogil
cdef void interval_t_vector_reset(interval_t_vector* vec) nogil

cdef void free_interval_t_vector(interval_t_vector* vec) nogil
cdef void print_interval_t_vector(interval_t_vector* vec) nogil

cdef class IntervalVector(object):
    cdef __cythonbufferdefaults__ = {'ndim' : 1, 'mode':'c'}

    cdef:
        interval_t_vector* impl
        int flags

    cdef int allocate_storage(self) nogil
    cdef int allocate_storage_with_size(self, size_t size) nogil

    cdef int free_storage(self) nogil
    cdef bint get_should_free(self) nogil
    cdef void set_should_free(self, bint flag) nogil

    cdef interval_t* get_data(self) nogil

    @staticmethod
    cdef IntervalVector _create(size_t size)

    @staticmethod
    cdef IntervalVector wrap(interval_t_vector* vector)

    cdef interval_t get(self, size_t i) nogil
    cdef void set(self, size_t i, interval_t value) nogil
    cdef size_t size(self) nogil
    cdef int cappend(self, interval_t value) nogil

    cdef IntervalVector _slice(self, object slice_spec)

    cpdef IntervalVector copy(self)

    cpdef int append(self, tuple value) except *
    cpdef int extend(self, object values) except *

    cpdef int reserve(self, size_t size) nogil

    cpdef int fill(self, interval_t value) nogil



    cpdef object _to_python(self, interval_t value)
    cpdef interval_t _to_c(self, object value) except *