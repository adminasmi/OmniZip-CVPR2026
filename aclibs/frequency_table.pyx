# Reference arithmetic coding
#
# Copyright (c) Project Nayuki
# MIT License. See readme file.
# https://www.nayuki.io/page/reference-arithmetic-coding
#

import numpy as np
from numba_helpers import calc_cumulative

cdef class FrequencyTable:
    cpdef int get_symbol_limit(self):
        raise NotImplementedError()

    cpdef int get(self, int symbol):
        raise NotImplementedError()

    cpdef void set(self, int symbol, int freq):
        raise NotImplementedError()

    cpdef void increment(self, int symbol):
        raise NotImplementedError()

    cpdef int get_total(self):
        raise NotImplementedError()

    cpdef int get_low(self, int symbol):
        raise NotImplementedError()

    cpdef int get_high(self, int symbol):
        raise NotImplementedError()


cdef class SimpleFrequencyTable(FrequencyTable):
    def __init__(self, freqs):
        # Ensure freqs is a numpy array with the correct dtype
        if not isinstance(freqs, np.ndarray):
            freqs = np.array(freqs, dtype=np.int32)
        elif freqs.dtype != np.int32:
            freqs = freqs.astype(np.int32)

        self.frequencies = freqs
        self.total = sum(freqs)
        self.cumulative = calc_cumulative(freqs)

    cpdef int get_symbol_limit(self):
        return len(self.frequencies)

    cpdef int get(self, int symbol):
        self._check_symbol(symbol)
        return self.frequencies[symbol]

    cpdef void set(self, int symbol, int freq):
        self._check_symbol(symbol)
        self.total += freq - self.frequencies[symbol]
        self.frequencies[symbol] = freq
        self._init_cumulative()

    cpdef void increment(self, int symbol):
        self._check_symbol(symbol)
        self.total += 1
        self.frequencies[symbol] += 1
        self._init_cumulative()

    cpdef int get_total(self):
        return self.total

    cpdef int get_low(self, int symbol):
        self._check_symbol(symbol)
        return self.cumulative[symbol - 1] if symbol > 0 else 0

    cpdef int get_high(self, int symbol):
        self._check_symbol(symbol)
        return self.cumulative[symbol]

    cdef void _init_cumulative(self):
        self.cumulative = calc_cumulative(self.frequencies)

    cdef void _check_symbol(self, int symbol):
        if symbol < 0 or symbol >= len(self.frequencies):
            raise ValueError("Symbol out of range")


cdef class CheckedFrequencyTable(FrequencyTable):
    def __init__(self, FrequencyTable freqtable):
        self.freqtable = freqtable

    cpdef int get_symbol_limit(self):
        return self.freqtable.get_symbol_limit()

    cpdef int get(self, int symbol):
        self._check_symbol(symbol)
        return self.freqtable.get(symbol)

    cpdef int get_total(self):
        return self.freqtable.get_total()

    cpdef int get_low(self, int symbol):
        self._check_symbol(symbol)
        return self.freqtable.get_low(symbol)

    cpdef int get_high(self, int symbol):
        self._check_symbol(symbol)
        return self.freqtable.get_high(symbol)

    cpdef void set(self, int symbol, int freq):
        self._check_symbol(symbol)
        self.freqtable.set(symbol, freq)

    cpdef void increment(self, int symbol):
        self._check_symbol(symbol)
        self.freqtable.increment(symbol)

    cdef bint _is_symbol_in_range(self, int symbol):
        return 0 <= symbol < self.get_symbol_limit()

    cdef void _check_symbol(self, int symbol):
        if not self._is_symbol_in_range(symbol):
            raise ValueError("Symbol out of range")