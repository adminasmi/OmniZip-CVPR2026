

import torch
import time
import numpy as np

import sys
sys.path.append('/home/zhaoy/OmniZip-CVPR2026/aclibs')

import bitstreams
import arithmetic_coder
from numba_helpers import get_freqs

ARITHMETIC_CODER_PRECISION = 32


# ====== 你的熵编码函数 ======

def _arithmetic_encode(logits, targets):
    probs = torch.softmax(logits, dim=-1)
    bitout = bitstreams.BitOutputStream()
    encoder = arithmetic_coder.ArithmeticEncoder(numbits=ARITHMETIC_CODER_PRECISION, bitout=bitout)
    
    for probs_batch, tokens_batch in zip(probs, targets.to(torch.uint8)):
        for prob, token in zip(probs_batch, tokens_batch):
            freqs_table = get_freqs(prob)
            encoder.write(freqs_table, token)
    
    encoder.finish()
    bitout.close()
    return float(bitout.get_size_in_bits())

# ====== 构造测试输入 ======

def generate_dummy_data(batch_size=8, seq_len=128, vocab_size=256):
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    return logits, targets

# ====== 测速函数（重复取平均） ======

def test_encode_speed(logits, targets, runs=10):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        bits = _arithmetic_encode(logits, targets)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time_ms = np.mean(times) * 1000
    bpp = bits / (logits.shape[0] * logits.shape[1])
    print(f"Average encoding time over {runs} runs: {avg_time_ms:.3f} ms")
    print(f"Output bits: {bits:.2f}, Bits per token: {bpp:.4f}")
    return avg_time_ms, bpp

# ====== 主函数入口 ======

if __name__ == "__main__":
    # 模拟输入
    logits, targets = generate_dummy_data(batch_size=8, seq_len=128, vocab_size=256)

    # 测试速度
    test_encode_speed(logits, targets, runs=10)
