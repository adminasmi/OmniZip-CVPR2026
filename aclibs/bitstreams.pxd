cimport numpy as cnp

ctypedef cnp.int32_t np_int32
ctypedef cnp.float32_t np_float32

cdef class BitInputStream:
    cdef:
        object input              # The underlying byte stream to read from
        int currentbyte           # The accumulated bits for the current byte, always in the range [0x00, 0xFF] or -1 if end of stream is reached
        int numbitsremaining      # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)

    cpdef int read(self)
    cpdef int read_no_eof(self)
    cpdef void close(self)


cdef class BitOutputStream:
    cdef:
        int currentbyte    # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        int numbitsfilled  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
        bytearray data     # store the compressed data (bytes)

    cpdef void write(self, int b)
    cpdef void close(self)
    cpdef void write_to_file(self, str file_path)
    cpdef int get_size_in_bits(self)
    cpdef int get_size_in_bytes(self)
    cpdef bytes get_data(self)






