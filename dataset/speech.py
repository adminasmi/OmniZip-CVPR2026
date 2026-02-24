import os
import math
import mmap
import random
import logging
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import read_word2id_dict
from dataset import collate_fn_test

        
class SpeechDataset(Dataset):
    '''
    Almost same as speechDataset
    '''
    def __init__(
        self, 
        args,
        word2id_dict, 
        mode='train', 
        logger=logging.getLogger('base')
    ) -> None:
        super().__init__()
        self.args = args
        self.logger = logger
        
        if isinstance(args.speech_dir, list):
            assert len(args.speech_dir) == 2, "when args.speech_dir is a list, it should compose two directories for train and test."
            idx = 0 if mode == 'train' else 1
            self.speech_dir = Path(args.speech_dir[idx]).absolute()
        elif isinstance(args.speech_dir, str):
            self.speech_dir = Path(args.speech_dir).absolute()
        else:
            raise ValueError(f"Invalid type for args.speech_dir: {type(args.speech_dir)}. Expected str or list.")
        
        self.num_speeches = getattr(args, 'num_speeches', -1)
        if args.debug or mode == 'eval':
            self.num_speeches = 10
        self.speech_paths = sorted(self.speech_dir.rglob('*.txt'))[:self.num_speeches]
        
        self.seq_len = getattr(args, 'seq_len', 1024)
        self.word2id_dict = word2id_dict
        self.mode    = mode
        self.debug   = getattr(args, 'debug',  False)
        self.padding = getattr(args, 'padding', True)
        self.padding_unit = getattr(args, 'padding_unit', 16)
        
        # Define special tokens
        self.start_id = word2id_dict.get('<speech>', 0)  # Use modal token instead of <s>
        self.unk_id   = word2id_dict.get('<unk>',  0)
        self.pad_id   = word2id_dict.get('<pad>',  0)
            
        # Precompute token counts per file
        self.file_token_counts = []
        self.ntokens = 0
        for speech_path in self.speech_paths:
            self._cal_ntokens_per_file(speech_path)
        
        self.nseqs = math.ceil(self.ntokens / self.seq_len)
        self.logger.info(f"Found {len(self.speech_paths)} files, total tokens ≈ {self.ntokens}, seqs={self.nseqs}")
        
        # Prefix-sum index for fast file location
        self.prefix_sums = np.cumsum([0] + self.file_token_counts)
        self._log_attributes(logger)
               
        
    def __len__(self):
        return self.nseqs
    
    
    def _find_file_index(self, global_idx):
        """Find which file contains the given global token index."""
        file_idx = np.searchsorted(self.prefix_sums, global_idx, side='right') - 1
        local_start = global_idx - self.prefix_sums[file_idx]
        return file_idx, local_start
    
    def _read_tokens_from_file(self, path, start, count):
        """Efficiently read `count` tokens starting from `start` in the file."""
        tokens = []
        read_so_far = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if start >= len(parts):
                    start -= len(parts)
                    continue
                slice_part = parts[start : start + (count - read_so_far)]
                tokens.extend(slice_part)
                read_so_far += len(slice_part)
                if read_so_far >= count:
                    break
                start = 0
        return tokens
    
    
    def __getitem__(self, index):        
        start_token = index * self.seq_len
        file_idx, local_start = self._find_file_index(start_token)

        sequence = []
        remain = self.seq_len
        # Collect tokens from consecutive files if necessary
        while remain > 0 and file_idx < len(self.speech_paths):
            tokens_part = self._read_tokens_from_file(self.speech_paths[file_idx], local_start, remain)
            sequence.extend(tokens_part)
            remain -= len(tokens_part)
            file_idx += 1
            local_start = 0
            if len(tokens_part) == 0:
                break
            
        # apply random augment
        if self.mode == 'train' and random.random() < 0.5:
            choice = random.choice(['reverse', 'drop', 'scale'])
            if choice == 'reverse':
                sequence = sequence[::-1]
            elif choice == 'drop':
                sequence = [t for t in sequence if random.random() > 0.05]
            elif choice == 'scale':
                sequence = sequence[:np.random.randint(low=int(self.seq_len*0.95), high=self.seq_len)]

        return self._tokenize_byte_uint8_sequence(sequence)

        
        # sequence = self.tokens[index * self.seq_len: min((index + 1) * self.seq_len, len(self.tokens))]
        # if random.random() < 0.5:
        #     choice = random.choice(['reverse', 'drop', 'scale'])
        #     if choice == 'reverse':
        #         sequence = sequence[::-1]
        #     elif choice == 'drop':
        #         sequence = [t for t in sequence if random.random() > 0.05]
        #     elif choice == 'scale':
        #         sequence = sequence[:np.random.randint(low=int(self.seq_len*0.95), high=self.seq_len)]
                
        # return self._tokenize_byte_uint8_sequence(sequence)
        
    
    def _cal_ntokens_per_file(self, speech_path):
        try:
            with open(speech_path, "r", encoding="utf-8") as f:
                count = sum(len(line.split()) for line in f)
                self.file_token_counts.append(count)
                self.ntokens += count
        except Exception as e:
            self.logger.warning(f"Skipped {speech_path} due to error: {e}")
        
        
    def _tokenize_byte_uint8_sequence(self, sequence):
        tokens  = [self.word2id_dict[s] for s in sequence]     
        targets = [int(s) for s in sequence]
        inputs  = [self.start_id] + tokens[:-1]
        
        seqlen = len(inputs)      # add padding
        pad_len = (self.padding_unit - (seqlen % self.padding_unit)) % self.padding_unit if self.padding else 0
        if self.padding and pad_len > 0:
            inputs  += [self.pad_id] * pad_len
            targets += [self.pad_id] * pad_len
        masks = [i != self.pad_id for i in inputs]
            
        assert max(targets) <= 255 and min(targets) >= 0
        return {
            'input_tokens':  torch.tensor(inputs, dtype=torch.long), 
            'target_tokens': torch.tensor(targets, dtype=torch.long),
            'masks':         torch.tensor(masks, dtype=torch.bool),
            'modality':      'speech',
        }
        
    def _log_attributes(self, logger=logging.getLogger('base')):
        """
        Log all the key attributes of the SpeechtDataset instance in a neat format.
        """
        logger.info(f"========== SpeechDataset Attributes ({self.mode.upper()}) ==========")
        logger.info(f"{'speech_dir':<22}: {self.speech_dir}")
        logger.info(f"{'seq_len':<22}: {self.seq_len}")
        logger.info(f"{'debug':<22}: {self.debug}")
        logger.info(f"{'num speech files':<22}: {self.num_speeches}")
        logger.info(f"{'padding':<22}: {self.padding}")
        logger.info(f"{'padding_unit':<22}: {self.padding_unit}")
        logger.info(f"{'tokens':<22}: {self.ntokens} total tokens")
        # if len(self.tokens) > 3:
        #     logger.info(f"{'':<22}  -> Sample: {self.tokens[:3]} ...")
        # else:
        #     logger.info(f"{'':<22}  -> {self.tokens}")
        if self.mode == 'test':
            logger.info(f"{'seg_tokens':<22}: {self.nseqs} total sentences")
        logger.info("=============================================\n")

    def visualize(self):
        '''
        Visualize the dataset structure.
        '''
        print('--- LibriSpeech Dataset Structure ---')
        
        # Get a sample and display info
        sample_idx = 0
        waveform, sample_rate, speech, speaker_id, chapter_id, utterance_id = self[sample_idx]
        
        print(f'Total samples: {len(self.dataset)}')
        print(f'Sample rate: {sample_rate} Hz')
        print(f'Example speech: \'{speech}\'')
        print(f'Speaker ID: {speaker_id}')
        print(f'Chapter ID: {chapter_id}')
        print(f'Utterance ID: {utterance_id}')
        print(f'Waveform shape: {waveform.shape} (channels, samples)')
        print(f'Duration: {waveform.shape[1]/sample_rate:.2f} seconds')


if __name__ == '__main__':
    from dataclasses import dataclass, field
    from typing import List
    @dataclass
    class ARGS:
        seq_len: int = 1024
        debug: bool  = True
        speech_dir: str = '/media/ps/ssd6/zhaoy/datasets/speech/LibriSpeech/conv2text_uint8/LibriSpeech-train-100-clean'
        vocab_dict: str = '/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json'
    args = ARGS()
    
    dataset  = SpeechDataset(
        args,
        word2id_dict=read_word2id_dict(args.vocab_dict),
        mode='test',
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, collate_fn=collate_fn_test)

    for idx, data in enumerate(dataloader):
        inputs  = data['input_tokens']
        targets = data['target_tokens']
        masks   = data['masks']
        print(f'{inputs}\n\n')
        print(f'{targets}')
        break

