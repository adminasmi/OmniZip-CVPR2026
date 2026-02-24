# Reference arithmetic coding
#
# Copyright (c) Project Nayuki
# MIT License. See readme file.
# https://www.nayuki.io/page/reference-arithmetic-coding
#

# ---- Bit-oriented I/O streams ----
import io
import torch
import cython
import numpy as np

from numba_helpers import process_probs, clip_freqs

# A stream of bits that can be read. Because they come from an underlying byte stream,
# the total number of bits is always a multiple of 8. The bits are read in big endian.
cdef class BitInputStream:

    # Constructs a bit input stream based on the given byte input stream.
    def __init__(self, inp):
        self.input = io.BytesIO(inp) if isinstance(inp, bytes) else inp     # The underlying byte stream to read from
        self.currentbyte = 0        # Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
        self.numbitsremaining = 0   # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)

    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or -1 if
    # the end of stream is reached. The end of stream always occurs on a byte boundary.
    cpdef int read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0]
            self.numbitsremaining = 8

        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1

    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or raises an EOFError
    # if the end of stream is reached. The end of stream always occurs on a byte boundary.
    cpdef int read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()

    # Closes this stream and the underlying input stream.
    cpdef void close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


# A stream where bits can be written to. Because they are written to an underlying
# byte stream, the end of the stream is padded with 0's up to a multiple of 8 bits.
# The bits are written in big endian.
cdef class BitOutputStream:
    # Constructs a bit output stream based on the given byte output stream.
    def __init__(self):
        self.currentbyte = 0        # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.numbitsfilled = 0      # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
        self.data = bytearray()     # store the compressed data (bytes)

    # Writes a bit to the stream. The given bit must be 0 or 1.
    cpdef void write(self, int b):
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")

        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1

        if self.numbitsfilled == 8:
            self.data.extend(bytes([self.currentbyte]))
            self.currentbyte = 0
            self.numbitsfilled = 0

    # Closes this stream and the underlying output stream. If called when this bit stream is not at a byte boundary,
    # then the minimum number of "0" bits (between 0 and 7 of them) are written as padding to reach the next byte boundary.
    cpdef void close(self):
        while self.numbitsfilled != 0:
            self.write(0)

    # ZY add - write the out stream to a file.
    cpdef void write_to_file(self, str file_path):
        with open(file_path, "wb") as f:
            f.write(self.data)

    # ZY add - get output's size in bit
    cpdef int get_size_in_bits(self):
        return len(self.data) * 8 + self.numbitsfilled

    # ZY add - get output's size in byte
    cpdef int get_size_in_bytes(self):
        return len(self.data) + (1 if self.numbitsfilled > 0 else 0)

    # ZY add - get output's data
    cpdef bytes get_data(self):
        self.close()
        return bytes(self.data)


