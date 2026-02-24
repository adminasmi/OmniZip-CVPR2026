from dataclasses import dataclass

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import torch
import torch.nn as nn
logger = logging.getLogger('base')

from registry import MODULE_BUILD_FUNCS
from rwkv7_hira_moa_moe import BaseRWKV_HIRA_MOA_MOE


@MODULE_BUILD_FUNCS.register_with_name(module_name='rwkv7_hira_moe')
def build_rwkv7_hira(args):
    from attnblks import AttnBlkHiRA
    model = BaseRWKV_HIRA_MOA_MOE(args, attn_blk_cls=AttnBlkHiRA, use_mlp_moe=True)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return model, criterion


if __name__ == '__main__':
    torch.cuda.set_device(0)
    
    @dataclass
    class ARGS:
        k: int = 2
        num_experts: int = 4
        
        mlp_factor: int = 4
        num_moe_layers: int = 2
        num_moa_layers: int = 2
        
        k_moe: int = 2
        k_moa: int = 2
        num_experts_moe: int = 4
        num_experts_moa: int = 4
        vocab_size : int = 16384
        n_layer : int = 2
        n_embd : int  = 320
        hira_factor: int = 4
        inference: bool  = True
    args = ARGS()
    print(args)
    
    batch_size = 2
    seq_len = 1024
    
    model, criterion = build_rwkv7_hira(args)
    model.count_params()
    
    model.switch_to_inference()
    model = model.cuda()
    
    indices = torch.randint(0, args.vocab_size, (1, 16)).cuda()
    dummy_inputs = (indices,)
    # flops, _ = profile(model, inputs=dummy_inputs)
    # print(f'FLOPs: {flops / (16 * 1024**2):.2f}M FLOPs.\n\n')
    # print(f'MACs: {flops / (2 * 16 * 1024**2):.2f}M MACs.\n\n')
    
    print('\nProfile by fvcore.')
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_inputs)
    print(f'FLOPs: {flops.total() / (16 * 1024**2):.2f}M FLOPs.\n')
    print(f'MACs: {flops.total() / (2 * 16 * 1024**2):.2f}M MACs.\n\n')