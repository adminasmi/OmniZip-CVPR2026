
cimport numpy as cnp
ctypedef cnp.int32_t np_int32

cdef class FrequencyTable:
    cpdef int get_symbol_limit(self)
    cpdef int get(self, int symbol)
    cpdef void set(self, int symbol, int freq)
    cpdef void increment(self, int symbol)
    cpdef int get_total(self)
    cpdef int get_low(self, int symbol)
    cpdef int get_high(self, int symbol)


cdef class SimpleFrequencyTable(FrequencyTable):
    cdef:
        cnp.ndarray frequencies
        int total
        cnp.ndarray cumulative

    cpdef int get_symbol_limit(self)
    cpdef int get(self, int symbol)
    cpdef void set(self, int symbol, int freq)
    cpdef void increment(self, int symbol)
    cpdef int get_total(self)
    cpdef int get_low(self, int symbol)
    cpdef int get_high(self, int symbol)
    cdef void _init_cumulative(self)
    cdef void _check_symbol(self, int symbol)


cdef class CheckedFrequencyTable(FrequencyTable):
    cdef FrequencyTable freqtable

    cpdef int get_symbol_limit(self)
    cpdef int get(self, int symbol)
    cpdef int get_total(self)
    cpdef int get_low(self, int symbol)
    cpdef int get_high(self, int symbol)
    cpdef void set(self, int symbol, int freq)
    cpdef void increment(self, int symbol)
    cdef bint _is_symbol_in_range(self, int symbol)
    cdef void _check_symbol(self, int symbol)