import torch
import numpy as np
from numba import njit
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frequency_table import SimpleFrequencyTable

@njit
def calc_cumulative_numba(freqs):
    length = freqs.shape[0]
    result = np.zeros(length, dtype=np.int32)
    sum_ = 0
    for i in range(length):
        sum_ += freqs[i]
        result[i] = sum_
    return result


def calc_cumulative(freqs):
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.cpu().numpy()

    result = calc_cumulative_numba(freqs)
    return result



@njit
def process_probs(c_probs, scale_factor):
    length = c_probs.shape[0]
    freqs = np.empty(length, dtype=np.int32)
    for i in range(length):
        freqs[i] = int(c_probs[i] * scale_factor)
    return freqs


@njit
def clip_freqs(freqs, min_value, max_value):
    for i in range(freqs.shape[0]):
        if freqs[i] < min_value:
            freqs[i] = min_value
        elif freqs[i] > max_value:
            freqs[i] = max_value
    return freqs


def get_freqs(probs, scale_factor:int = 1<<24):
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    elif isinstance(probs, list):
        probs = np.array(probs, dtype=np.float32)
    elif isinstance(probs, np.ndarray):
        if probs.dtype != np.float32:
            probs = probs.astype(np.float32)
    else:
        raise TypeError("Input should be a list, numpy array, or torch tensor")

    freqs = process_probs(probs, scale_factor)
    freqs = clip_freqs(freqs, 1, freqs.max())

    return SimpleFrequencyTable(freqs.tolist())