#!/bin/bash
# test AC/DC

# => expected output:
# Testing direct arithmetic coding...
# Original symbols: [5, 0, 0, 6, 9, 7, 1, 0, 9, 9, 0, 9, 4, 1, 8, 9, 4, 6, 4, 3]
# Compressed size: 8 bytes
# Decoded symbols: [5, 0, 0, 6, 9, 7, 1, 0, 9, 9, 0, 9, 4, 1, 8, 9, 4, 6, 4, 3]
# ✓ Perfect reconstruction!

python -c "
import torch
import numpy as np
from frequency_table import SimpleFrequencyTable, CheckedFrequencyTable
from bitstreams import BitInputStream, BitOutputStream
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder

# Test direct arithmetic coding
print('Testing direct arithmetic coding...')

vocab_size = 10
num_symbols = 20

# Create frequency table
freqs = np.array([10, 5, 8, 3, 12, 7, 9, 4, 6, 11], dtype=np.int32)
simple_freq_table = SimpleFrequencyTable(freqs)
freq_table = CheckedFrequencyTable(simple_freq_table)

# Generate symbols
symbols = []
cumsum = np.cumsum(freqs)
total = cumsum[-1]
for i in range(num_symbols):
    rand_val = np.random.randint(0, total)
    symbol = np.searchsorted(cumsum, rand_val)
    symbols.append(symbol)

print(f'Original symbols: {symbols}')

# Encode
bitout = BitOutputStream()
encoder = ArithmeticEncoder(numbits=32, bitout=bitout)

for symbol in symbols:
    encoder.write(freq_table, symbol)

encoder.finish()
bitout.close()

compressed_data = bitout.get_data()
print(f'Compressed size: {len(compressed_data)} bytes')

# Decode
bitin = BitInputStream(compressed_data)
decoder = ArithmeticDecoder(numbits=32, bitin=bitin)

decoded_symbols = []
for i in range(num_symbols):
    try:
        symbol = decoder.read(freq_table)
        decoded_symbols.append(symbol)
    except Exception as e:
        print(f'Decoding error at position {i}: {e}')
        break

bitin.close()

print(f'Decoded symbols: {decoded_symbols}')

# Compare
if symbols == decoded_symbols:
    print('✓ Perfect reconstruction!')
else:
    print('✗ Reconstruction failed!')
    print(f'Expected: {symbols}')
    print(f'Got: {decoded_symbols}')
"