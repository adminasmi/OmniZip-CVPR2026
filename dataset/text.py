#-- Check Done --#
import os
import logging

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import regex as re
import mmap
import numpy as np
from abc import ABC

import torch
from torch.utils.data import Dataset

def get_size(path):
    """Return size (in bytes) of a file or directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    else:
        raise FileNotFoundError(f"Path not found: {path}")

AUGMENT = 500
class TextBaseDataset(Dataset, ABC):
    def __init__(
        self, args, word2id_dict, mode='train', logger=None
    ):
        super().__init__()
        self.args = args
        self.logger = logger or logging.getLogger('base')
        self.mode = mode
        self.word2id_dict = word2id_dict
        self.seq_len = getattr(args, 'seq_len', 1024)
        
        self.padding_unit = getattr(args, 'padding_unit', 16)
        self.padding = getattr(args, 'padding', True)
        self.debug = getattr(args, 'debug', False)
        self.epoch  = 0
        self.nsteps = 0
        
        # Define special tokens
        # self.start_id = word2id_dict.get('<s>',    0)
        self.start_id = word2id_dict.get('<text>', 0)  # Use modal token instead of <s>
        self.unk_id   = word2id_dict.get('<unk>',  0)
        self.pad_id   = word2id_dict.get('<pad>',  0)
        
        self.modality = 'text'
        
        self._resolve_paths()
        self._load_corpus()
        
    def __len__(self):
        return self.nsteps
    
    def _resolve_paths(self):
        path_configs = [
            (self.args.text_file, 'text_path'),
            (self.args.text_corpus, 'corpus_path'),
        ]
        for paths, attr in path_configs:
            if isinstance(paths, list):
                assert len(paths) == 2, f"{attr.replace('_', ' ')} requires 2 paths for train/test"
                path = paths[0 if self.mode == 'train' else 1]
            else:
                path = paths
            setattr(self, attr, path)
               
    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            self.logger.error(f'Do not exist corpus file: {self.corpus_path}.')
            raise FileNotFoundError
        
        self.logger.info(f'Loading tokens from {self.corpus_path}...')
        if self.debug or self.mode == 'eval':
            self.ntokens = int(1e6)
        else:
            self.ntokens = -1
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            if self.debug or self.mode == 'eval':
                chunk = f.read(int(10 * 1e6))  # 10MB
                self.tokens = list(map(int, re.findall(r'\d+', chunk)))[:self.ntokens]
            else:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                data = mm.read().decode('utf-8')
                mm.close()
                self.tokens = list(map(int, re.findall(r'\d+', data)))
        self.tokens = self.tokens[:self.ntokens]
        self.logger.info('Done!')
        
        # raw bytes
        if self.debug or self.mode == 'eval':
            self.raw_bytes = self.ntokens * 3.29    # approx
        else:
            self.raw_bytes = get_size(self.text_path)
        
        # nsteps
        max_start = max(0, len(self.tokens) - self.seq_len)
        if self.mode == 'train':
            self.step_len = self.seq_len // AUGMENT
            self.nsteps = max_start // self.step_len
        else:
            self.step_len = self.seq_len
            self.nsteps = max(1, len(self.tokens) // self.step_len)

    def _log_attributes(self, logger=logging.getLogger('base')):
        """
        Log all the key attributes of the SQLDataset instance in a neat format.
        """
        logger.info(f"========== {self.modality[0].upper() + self.modality[1:].lower()}Dataset Attributes ({self.mode.upper()}) ==========")
        logger.info(f"{'corpus_path':<22}: {self.corpus_path}")
        logger.info(f"{'text_path':<22}: {self.text_path}")
        logger.info(f"{'seq_len':<22}: {self.seq_len}")
        logger.info(f"{'debug':<22}: {self.debug}")
        logger.info(f"{'padding':<22}: {self.padding}")
        logger.info(f"{'padding_unit':<22}: {self.padding_unit}")
        logger.info(f"{'number of raw bytes':<22}: {self.raw_bytes}")
        logger.info(f"{'tokens':<22}: {len(self.tokens)} total tokens")
        if len(self.tokens) > 3:
            logger.info(f"{'':<22}  -> Sample: {self.tokens[:3]} ...")
        else:
            logger.info(f"{'':<22}  -> {self.tokens}")
        logger.info(f"{'seg_tokens':<22}: {self.nsteps} total sentences")
        logger.info("=============================================\n")
        
        
    def _set_epoch(self, epoch):
        self.epoch = epoch
        
    def _augment(self, tokens, rng):
        if rng.random() < 0.6:
                choice = rng.choice(['reverse', 'drop', 'scale'], p=[0.2, 0.4, 0.4])
                if choice == 'reverse':
                    tokens = tokens[::-1]
                elif choice == 'drop':
                    mask = rng.random(len(tokens)) > 0.05
                    tokens = list(np.array(tokens)[mask])
                elif choice == 'scale':
                    new_len = rng.integers(int(self.seq_len * 0.9), self.seq_len)
                    tokens  = tokens[:new_len]
        return tokens
        
    def __getitem__(self, index):
        '''
        Generates a batch of input and target tokens with types.

        Returns:
            instance (dict): Contains input_tokens, target_tokens, input_types, and target_types as tensors.
        '''
        rng = np.random.default_rng(seed=index + self.epoch * 1024)     # random seed
        start = index * self.step_len
        
        tokens = self.tokens[start : start + self.seq_len]
        if self.mode == 'train':
            tokens = self._augment(tokens, rng)
        
        pad_len = (self.padding_unit - (len(tokens) % self.padding_unit)) % self.padding_unit
        target_tokens = tokens + [self.pad_id] * pad_len
        input_tokens  = [self.start_id] + target_tokens[:-1]
        masks = [t != self.pad_id for t in input_tokens]
        
        instance = {
            'input_tokens':  torch.tensor(input_tokens, dtype=torch.long),
            'target_tokens': torch.tensor(target_tokens, dtype=torch.long),
            'masks':         torch.tensor(masks, dtype=torch.bool),
            'modality':      self.modality,
        }
        return instance

class TextDataset(TextBaseDataset):
    def __init__(
        self, args, word2id_dict, mode='train', logger=None
    ):
        super().__init__(args, word2id_dict, mode, logger)
        self.modality = 'text'
        self._log_attributes()
        
        
if __name__ == '__main__':
    from dataclasses import dataclass, field
    from typing import List
    from dataset import collate_fn_test
    from utils.common import read_word2id_dict
    from torch.utils.data import DataLoader
    
    UNK_TYPE    = 'unk_allow'
    VOCAB_SIZE  = 16384
    COVERAGE    = 1.0
    
    @dataclass
    class ARGS:
        debug : bool = False
        seq_len : int = 1024
        corpus_root: str = f'./corpus/{UNK_TYPE}'
        text_file  : List[str] = field(default_factory=lambda: ['text/enwik8', 'text/enwik8'])
        text_corpus: List[str] = field(default_factory=lambda: [f'spm_enwik8_bpe_{VOCAB_SIZE}_{COVERAGE}.txt', f'spm_enwik8_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'])
        text_unk  : List[str] = field(default_factory=lambda: [f'unk_enwik8_bpe_{VOCAB_SIZE}_{COVERAGE}.txt', f'unk_enwik8_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'])
    args = ARGS()
    
    dataset = TextDataset(
        args=args,
        mode='train',
        word2id_dict=read_word2id_dict(f'/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_{VOCAB_SIZE}_{COVERAGE}.json')
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    for idx, data in enumerate(dataloader):
        print(data['input_tokens'])
        print(data['input_tokens'].shape)
        break