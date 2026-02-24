#cython: language_level=3

# Reference arithmetic coding
#
# Copyright (c) Project Nayuki
# MIT License. See readme file.
# https://www.nayuki.io/page/reference-arithmetic-coding
#
cimport frequency_table
cimport bitstreams
from libc.stdint cimport uint64_t, uint32_t

# ---- Arithmetic coding core classes ----

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
cdef class ArithmeticCoderBase:
    cdef:
        int num_state_bits
        uint64_t full_range, half_range, quarter_range, minimum_range, maximum_total, state_mask
        uint64_t low, high

    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, int numbits):
        if numbits < 1:
            raise ValueError("State size out of range")

        # -- Configuration fields --
        self.num_state_bits = numbits
        self.full_range = <uint64_t>1 << self.num_state_bits
        self.half_range = self.full_range >> 1  # Non-zero
        self.quarter_range = self.half_range >> 1  # Can be zero
        self.minimum_range = self.quarter_range + 2  # At least 2
        self.maximum_total = self.minimum_range
        self.state_mask = self.full_range - 1

        # -- State fields --
        self.low = 0
        self.high = self.state_mask

    # Updates the code range (low and high) of this arithmetic coder as a result
    # of processing the given symbol with the given frequency table.
    cdef void update(self, frequency_table.CheckedFrequencyTable freqs, uint32_t symbol):
        # State check
        cdef uint64_t low = self.low
        cdef uint64_t high = self.high
        cdef uint64_t range, newlow, newhigh

        if low >= high or (low & self.state_mask) != low or (high & self.state_mask) != high:
            raise AssertionError("Low or high out of range")

        range = high - low + 1
        if not (self.minimum_range <= range <= self.full_range):
            raise AssertionError("Range out of range")

        # Frequency table values check
        cdef uint32_t total = freqs.get_total()
        cdef uint32_t symlow = freqs.get_low(symbol)
        cdef uint32_t symhigh = freqs.get_high(symbol)
        if symlow == symhigh:
            raise ValueError("Symbol has zero frequency")
        if total > self.maximum_total:
            raise ValueError("Cannot code symbol because total is too large")

        # Update range
        newlow  = low + symlow * range // total
        newhigh = low + symhigh * range // total - 1
        self.low  = newlow
        self.high = newhigh

        # While low and high have the same top bit value, shift them out
        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = ((self.low << 1) & self.state_mask)
            self.high = ((self.high << 1) & self.state_mask) | 1
        # Now low's top bit must be 0 and high's top bit must be 1

        # While low's top two bits are 01 and high's are 10, delete the second highest bit of both
        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    cdef void shift(self):
        raise NotImplementedError()

    # Called to handle the situation when low=01(...) and high=10(...).
    cdef void underflow(self):
        raise NotImplementedError()


# Encodes symbols and writes to an arithmetic-coded bit stream.
cdef class ArithmeticEncoder(ArithmeticCoderBase):
    cdef:
        bitstreams.BitOutputStream output
        uint32_t num_underflow

    # Constructs an arithmetic coding encoder based on the given bit output stream.
    def __init__(self, int numbits, bitstreams.BitOutputStream bitout):
        super(ArithmeticEncoder, self).__init__(numbits)
        # The underlying bit output stream.
        self.output = bitout
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0

    # Encodes the given symbol based on the given frequency table.
    # This updates this arithmetic coder's state and may write out some bits.
    cpdef void write(self, frequency_table.FrequencyTable freqs, uint32_t symbol):
        if not isinstance(freqs, frequency_table.CheckedFrequencyTable):
            freqs = frequency_table.CheckedFrequencyTable(freqs)
        self.update(freqs, symbol)

    # Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
    # It is important that this method must be called at the end of the each encoding process.
    # Note that this method merely writes data to the underlying output stream but does not close it.
    cpdef void finish(self):
        self.output.write(1)

    cdef void shift(self):
        cdef int bit = self.low >> (self.num_state_bits - 1)
        self.output.write(bit)

        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0

    cdef void underflow(self):
        self.num_underflow += 1



# Reads from an arithmetic-coded bit stream and decodes symbols.
cdef class ArithmeticDecoder(ArithmeticCoderBase):
    cdef:
        object input
        uint64_t code

    # Constructs an arithmetic coding decoder based on the
    # given bit input stream, and fills the code bits.
    def __init__(self, int numbits, object bitin):
        super(ArithmeticDecoder, self).__init__(numbits)
        # The underlying bit input stream.
        self.input = bitin
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        for _ in range(self.num_state_bits):
            self.code = self.code << 1 | self.read_code_bit()

    # Decodes the next symbol based on the given frequency table and returns it.
    # Also updates this arithmetic coder's state and may read in some bits.
    cpdef unsigned int read(self, frequency_table.CheckedFrequencyTable freqs):
        cdef unsigned int total, start, end, middle, symbol
        cdef uint64_t range, offset, value

        if not isinstance(freqs, frequency_table.CheckedFrequencyTable):
            freqs = frequency_table.CheckedFrequencyTable(freqs)

        total = freqs.get_total()
        if total > self.maximum_total:
            raise ValueError("Cannot decode symbol because total is too large")

        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
        assert value * range // total <= offset
        assert 0 <= value < total

        start = 0
        end = freqs.get_symbol_limit()
        while end - start > 1:
            middle = (start + end) >> 1
            if freqs.get_low(middle) > value:
                end = middle
            else:
                start = middle
        assert start + 1 == end

        symbol = start
        assert freqs.get_low(symbol) * range // total <= offset < freqs.get_high(symbol) * range // total
        self.update(freqs, symbol)
        if not (self.low <= self.code <= self.high):
            raise AssertionError("Code out of range")
        return symbol

    cdef void shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

    cdef void underflow(self):
        self.code = (self.code & self.half_range) | ((self.code << 1) & (self.state_mask >> 1)) | self.read_code_bit()

    # Returns the next bit (0 or 1) from the input stream. The end
    # of stream is treated as an infinite number of trailing zeros.
    cdef int read_code_bit(self):
        cdef int temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp
