cimport cython

from libc.stdlib cimport malloc, realloc, free
from libc cimport *

from cpython.exc cimport PyErr_BadArgument
from cpython.mem cimport PyObject_Malloc, PyObject_Free

from cpython.sequence cimport (
    PySequence_Size, PySequence_Check, PySequence_Fast,
    PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE)
from cpython.slice cimport PySlice_GetIndicesEx

cdef enum VectorStateEnum:
    should_free = 1

cdef interval_t interval_from_tuple(tuple pair):
    cdef:
        interval_t value

    value.start = pair[0]
    value.end = pair[1]
    return value


cdef tuple tuple_from_interval(interval_t interval):
    return (interval.start, interval.end)



cdef extern from * nogil:
    int printf (const char *template, ...)
    void qsort (void *base, unsigned short n, unsigned short w, int (*cmp_func)(void*, void*))


DEF GROWTH_RATE = 2
DEF INITIAL_SIZE = 4


cdef int initialize_interval_vector_t(interval_t_vector* vec, size_t size) nogil:
    vec.v = <interval_t*>malloc(sizeof(interval_t) * size)
    vec.size = size
    vec.used = 0
    if vec.v == NULL:
        vec.size = 0
        return 1
    return 0


cdef interval_t_vector* make_interval_t_vector_with_size(size_t size) nogil:
    cdef:
        interval_t_vector* vec

    vec = <interval_t_vector*>malloc(sizeof(interval_t_vector))
    initialize_interval_vector_t(vec, size)

    return vec


cdef interval_t_vector* make_interval_t_vector() nogil:
    return make_interval_t_vector_with_size(INITIAL_SIZE)


cdef int interval_t_vector_resize(interval_t_vector* vec) nogil:
    cdef:
        size_t new_size
        interval_t* v
    new_size = vec.size * GROWTH_RATE
    v = <interval_t*>realloc(vec.v, sizeof(interval_t) * new_size)
    if v == NULL:
        printf("interval_t_vector_resize returned -1\n")
        return -1
    vec.v = v
    vec.size = new_size
    return 0


cdef int interval_t_vector_append(interval_t_vector* vec, interval_t value) nogil:
    if (vec.used + 1) >= vec.size:
        interval_t_vector_resize(vec)
    vec.v[vec.used] = value
    vec.used += 1
    return 0


cdef void free_interval_t_vector(interval_t_vector* vec) nogil:
    free(vec.v)
    free(vec)


cdef void print_interval_t_vector(interval_t_vector* vec) nogil:
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


cdef void interval_t_vector_reset(interval_t_vector* vec) nogil:
    vec.used = 0


cdef int interval_t_vector_reserve(interval_t_vector* vec, size_t new_size) nogil:
    cdef:
        interval_t* v
    v = <interval_t*>realloc(vec.v, sizeof(interval_t) * new_size)
    if v == NULL:
        printf("interval_t_vector_resize returned -1\n")
        return -1
    vec.v = v
    vec.size = new_size
    if new_size > vec.used:
        vec.used = new_size
    return 0





@cython.final
@cython.freelist(512)
cdef class IntervalVector(object):

    @staticmethod
    cdef IntervalVector _create(size_t size):
        cdef:
            IntervalVector self
        self = IntervalVector.__new__(IntervalVector)
        self.flags = 0
        self.allocate_storage_with_size(size)
        return self

    @staticmethod
    cdef IntervalVector wrap(interval_t_vector* vector):
        cdef:
            IntervalVector self
        self = IntervalVector.__new__(IntervalVector)
        self.flags = 0
        self.impl = vector
        self.set_should_free(False)
        return self

    def __init__(self, seed=None):
        cdef:
            size_t n
        self.flags = 0
        if seed is not None:
            if not PySequence_Check(seed):
                raise TypeError("Must provide a Sequence-like object")
            n = len(seed)
            self.allocate_storage_with_size(n)
            self.extend(seed)
        else:
            self.allocate_storage()

    cdef int allocate_storage_with_size(self, size_t size) nogil:
        if self.impl != NULL:
            if self.flags & VectorStateEnum.should_free:
                free_interval_t_vector(self.impl)
        self.impl = make_interval_t_vector_with_size(size)
        self.flags |= VectorStateEnum.should_free
        return self.impl == NULL

    cdef int allocate_storage(self) nogil:
        if self.impl != NULL:
            if self.flags & VectorStateEnum.should_free:
                free_interval_t_vector(self.impl)
        self.impl = make_interval_t_vector()
        self.flags |= VectorStateEnum.should_free
        return self.impl == NULL

    cdef int free_storage(self) nogil:
        free_interval_t_vector(self.impl)

    cdef bint get_should_free(self) nogil:
        return self.flags & VectorStateEnum.should_free

    cdef void set_should_free(self, bint flag) nogil:
        self.flags &= VectorStateEnum.should_free * flag

    cdef interval_t* get_data(self) nogil:
        return self.impl.v

    cdef interval_t get(self, size_t i) nogil:
        return self.impl.v[i]

    cdef void set(self, size_t i, interval_t value) nogil:
        self.impl.v[i] = value

    cdef size_t size(self) nogil:
        return self.impl.used

    cdef int cappend(self, interval_t value) nogil:
        return interval_t_vector_append(self.impl, value)

    cpdef int append(self, tuple value) except *:
        cdef:
            interval_t cvalue
        cvalue = self._to_c(value)
        return self.cappend(cvalue)

    cpdef int extend(self, object values) except *:
        cdef:
            size_t i, n
            object fast_seq
        if not PySequence_Check(values):
            raise TypeError("Must provide a Sequence-like object")

        fast_seq = PySequence_Fast(values, "Must provide a Sequence-like object")
        n = PySequence_Fast_GET_SIZE(values)

        for i in range(n):
            if self.append(<object>PySequence_Fast_GET_ITEM(fast_seq, i)) != 0:
                return 1

    cpdef int reserve(self, size_t size) nogil:
        return interval_t_vector_reserve(self.impl, size)

    cpdef int fill(self, interval_t value) nogil:
        cdef:
            size_t i, n
        n = self.size()
        for i in range(n):
            self.set(i, value)
        return 0

    cpdef IntervalVector copy(self):
        cdef:
            IntervalVector dup
            size_t i, n
        n = self.size()
        dup = IntervalVector._create(n)
        for i in range(n):
            dup.cappend(self.get(i))
        return dup

    cdef IntervalVector _slice(self, object slice_spec):
        cdef:
            IntervalVector dup
            Py_ssize_t length, start, stop, step, slice_length, i
        PySlice_GetIndicesEx(
            slice_spec, self.size(), &start, &stop, &step, &slice_length)
        dup = IntervalVector._create(slice_length)
        i = start
        while i < stop:
            dup.cappend(self.get(i))
            i += step
        return dup

    def __dealloc__(self):
        if self.get_should_free():
            self.free_storage()

    def __len__(self):
        return self.size()

    def __iter__(self):
        cdef:
            size_t i, n
        n = self.size()
        for i in range(n):
            yield self._to_python(self.get(i))

    def __getitem__(self, i):
        cdef:
            Py_ssize_t index
            size_t n
        if isinstance(i, slice):
            return self._slice(i)
        index = i
        n = self.size()
        if index < 0:
            index = n + index
        if index > n or index < 0:
            raise IndexError(index)
        return self._to_python(self.get(index))

    def __setitem__(self, i, value):
        cdef:
            Py_ssize_t index
            size_t n
        if isinstance(i, slice):
            raise TypeError("Does not support slice-assignment yet")
        n = self.size()
        index = i
        if index < 0:
            index = n + index
        if index > self.size() or index < 0:
            raise IndexError(i)
        self.set(index, self._to_c(value))

    def __repr__(self):
        return "{self.__class__.__name__}({members})".format(self=self, members=list(self))





    cpdef object _to_python(self, interval_t value):
        return tuple_from_interval(value)

    cpdef interval_t _to_c(self, object value) except *:
        return interval_from_tuple(value)