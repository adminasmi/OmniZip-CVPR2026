import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random


from collections import defaultdict
from typing import Iterator
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, mod_counts, batch_size, shuffle_within_batch=True) -> None:
        '''
        dtype_counts: dict {modality: count}
        batch_size: sum of all modalities' batch sizes.
        '''
        super().__init__()
        self.modalities = list(mod_counts.keys())
        self.mod_counts = mod_counts
        self.batch_size = batch_size
        self.shuffle_within_batch = shuffle_within_batch
        
        self.mod_starts = {}    # start index for each modality
        start = 0
        for mod, count in mod_counts.items():
            self.mod_starts[mod] = start
            start += count
        total_samples = sum(mod_counts.values())
        self.total_batches = total_samples // batch_size
    
    def __iter__(self) -> Iterator:
        mod_indices = {}
        for mod in self.modalities:
            start = self.mod_starts[mod]
            count = self.mod_counts[mod]
            indices = list(range(start, start + count))
            if self.shuffle_within_batch:
                random.shuffle(indices)
            mod_indices[mod] = self._cycle_indices(indices)
        
        mod_batch_size = self.batch_size // len(self.modalities)
        remaining = self.batch_size - mod_batch_size * len(self.modalities)
            
        for _ in range(self.total_batches):
            batch = []
            for mod in list(self.modalities):
                batch.extend(
                    [next(mod_indices[mod]) for _ in range(mod_batch_size)]
                )
            for _ in range(remaining):
                mod = random.choice(self.modalities)
                batch.append(next(mod_indices[mod]))
            if self.shuffle_within_batch:
                random.shuffle(batch)
            yield batch
        
    def _cycle_indices(self, indices):
        i = 0
        while True:
            yield indices[i]
            i = (i + 1) % len(indices)
    
    def __len__(self):
        return self.total_batches
    

def collate_fn_train(batch):
    # Step 1: Separate data of different modalities
    mod_data = defaultdict(lambda: defaultdict(list))
    for sample in batch:
        mod = sample['modality']
        for key in ['input_tokens', 'target_tokens', 'masks']:
            mod_data[mod][key].append(sample[key])

    # Step 2: Pad and stack data
    def pad_and_stack(tensors, pad_value=0):
        max_len = max(tensor.size(0) for tensor in tensors)
        padded_tensors = [
            torch.nn.functional.pad(tensor, (0, max_len - tensor.size(0)), value=pad_value)
            for tensor in tensors
        ]
        return torch.stack(padded_tensors)

    # Step 3: Process data of different modalities
    result = {}
    for mod, data in mod_data.items():                       
        result[mod] = {
            k: pad_and_stack(v, pad_value=False if k=='masks' else 0) for k, v in data.items()
        }
    return result


def collate_fn_test(batch):
    input_tokens = [item['input_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    masks = [item['masks'] for item in batch]

    max_len = max([x.shape[0] for x in input_tokens])

    def pad_tensor(tensor, max_len, pad_value=0):
        pad_size = max_len - tensor.shape[0]
        return torch.cat([tensor, torch.full((pad_size,), pad_value, dtype=tensor.dtype)])

    input_tokens = torch.stack([pad_tensor(x, max_len, pad_value=0) for x in input_tokens])
    target_tokens = torch.stack([pad_tensor(x, max_len, pad_value=0) for x in target_tokens])
    masks = torch.stack([pad_tensor(x, max_len, pad_value=False) for x in masks])

    return {
        'input_tokens': input_tokens,
        'target_tokens': target_tokens,
        'masks': masks
    }
