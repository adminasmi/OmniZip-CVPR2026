import os
import json
import torch


def extract_byte_uint8_values(vocab_file, begin_value=0, end_value=255) -> list[int]:
    '''
    Extract values from 'model' -> 'vocab' in the specified JSON file for keys '0' to '255', and return a list.
    '''
    assert os.path.isfile(vocab_file)
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_extract = []
    for i in range(begin_value, end_value + 1):
        value = vocab.get(str(i))
        vocab_extract.append(value)
        
    return vocab_extract


def extract_byte_latin1_values(vocab_file, begin_char= 0, end_char=255) -> list[int]:
    '''
    Extract values from 'model' -> 'vocab' in the specified JSON file for keys 'U+0000' to 'U+00FF' (i.e., Byte 0~255), and return a list.
    '''
    assert os.path.isfile(vocab_file)
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_extract = []
    for tid_str, piece in vocab.items():
        if len(tid_str) == 1:
            code = ord(tid_str)
            if begin_char <= code <= end_char:
                vocab_extract.append(int(piece))
    vocab_extract.sort()
    return vocab_extract


def sample_logits(logits, vocab_file):
    logits_clone = logits.clone()
    
    vocab = torch.load(vocab_file, map_location=logits_clone.device)    
    vocab = vocab.view(1, 1, vocab.size(0))
    
    indices = vocab.expand(logits_clone.size(0), logits_clone.size(1), vocab.size(-1))
    sampled_logits = torch.gather(logits_clone, dim=2, index=indices)        # (Batch, Seqlen, 256)
    return sampled_logits


if __name__ == '__main__':
    vocab_size = 16384
    coverage   = 1.0
    unk_type   = 'unk_allow'
    vocab_dir = f'/home/zhaoy/OmniZip-CVPR2026/vocabs/{unk_type}/vocab_spm_bpe_{vocab_size}_{coverage}'
    vocab_file = os.path.join(vocab_dir, f'spm_bpe_{vocab_size}_{coverage}.json')
    
    mode = 'byte_uint8'
    if mode == 'byte_uint8':
        vocab_extract = extract_byte_uint8_values(vocab_file)
    elif mode == 'byte_latin1':
        vocab_extract = extract_byte_latin1_values(vocab_file)
        
    vocab = torch.tensor(vocab_extract, dtype=torch.long)
    narrow_vocab_file = os.path.join(vocab_dir, f'spm_bpe_{mode}_255.model')
    torch.save(vocab, narrow_vocab_file)
    
    logits = torch.randn(1, 1, 128256)
    logits = sample_logits(logits=logits, vocab_file=narrow_vocab_file)
    
    print(logits.shape)
    print(logits)