########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from dataclasses import dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam

from registry import MODULE_BUILD_FUNCS
from models.common import RMSNorm, MLP
from attnblks import AttnBlkHiRA
    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.inference = args.inference
        self.attn = AttnBlkHiRA(args, layer_id=layer_id)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        
        self.mlp = MLP(n_embd=args.n_embd, factor=8)
        
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(self.ln1(x), v1)
        x = x + x1
        
        x2 = self.mlp(self.ln2(x))
        x = x + x2
        return x, v1
    
    def switch_to_inference(self):
        self.attn.merge_branches()        
        self.inference = True
        
    
class RWKV7_HIRA(nn.Module):
    def __init__(self, args, hira_factor=4):
        super().__init__()
        self.inference = args.inference
        self.rwkv = nn.ModuleDict(
            dict(
                wte = nn.Embedding(args.vocab_size, args.n_embd),
                h = nn.ModuleList([Block(args, layer_id) for layer_id in range(args.n_layer)]),
            )
        )
        self.rms_norm = RMSNorm(normalized_shape=(args.n_embd, )).cuda()
        self.lm_head  = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        
    def forward(self, idx):
        x = self.rwkv.wte(idx)
        x = self.rms_norm(x)
        x0 = x
        v1 = None
        
        for block in self.rwkv.h:
            x, v1 = block(x, v1, x0)
            assert x.isfinite().all() and v1.isfinite().all(), '`x` contains NaN or Inf.'
        x = self.rms_norm(x)
        logits = self.lm_head(x).float()
        return logits
        
    def configure_optimizers(self, args):
        no_decay = set()
        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)
                
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0},
        ]
        try:
            optimizer = FusedAdam(optim_groups, lr=args.lr, betas=args.betas, eps=args.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.Adam(optim_groups, lr=args.lr, betas=args.betas, eps=args.eps)

        return optimizer
    
    def switch_to_inference(self):
        for block in self.rwkv.h:
            block.switch_to_inference()
            
    def count_params(self):
        excluded_layers = ['key_A', 'key_B', 'value_A', 'value_B', 'receptance_A', 'receptance_B', 'wte', 'lm_head']
        nparams = 0
        for name, param in self.named_parameters():
            if not any(excl in name for excl in excluded_layers):
                nparams += param.numel()            
        print(f'--->>> num of params: {nparams/1024}K <<<---\n\n')
    
        
@MODULE_BUILD_FUNCS.register_with_name(module_name='rwkv7_hira')
def build_rwkv7_hira(args):
    model = RWKV7_HIRA(args)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return model, criterion
        

if __name__ == '__main__':
    torch.cuda.set_device(3)
    print(MODULE_BUILD_FUNCS.module_dict)
    @dataclass
    class ARGS:
        vocab_size : int = 16384
        n_layer : int = 3
        hira_factor : int = 4
        # n_head : int = 16    # head dim 128 suggested by @Grad62304977
        n_embd : int = 1488
    args = ARGS()
    print(args)
    
    args.inference = True
    model = RWKV7_HIRA(args).cuda().eval()
    model.count_params()
    model.switch_to_inference()
    model = model.cuda()
    
    indices = torch.randint(0, args.vocab_size, (1, 16)).cuda()
    dummy_inputs = (indices,)
    print('\nProfile by fvcore.')
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_inputs)
    print(f'FLOPs: {flops.total() / (16 * 1024**2):.2f}M FLOPs.\n')
    print(f'MACs: {flops.total() / (2 * 16 * 1024**2):.2f}M MACs.\n\n')
    
    # print(f'------> Start to test inference speed: {args.n_embd}e_{args.n_layer}l.\n\n')
    # batch_sizes = [1, 16, 128, 512, 1024, 2048, 3072]
    # seq_len = 256
    # for batch_size in batch_sizes:
    #     idx = torch.randint(0, args.vocab_size, (batch_size, seq_len), device='cuda')
    #     for _ in range(5):
    #         with torch.no_grad():
    #             _ = model(idx)
                
    #     n_repeat = 1000
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     with torch.no_grad():
    #         for _ in range(n_repeat):
    #             _ = model(idx)
    #     torch.cuda.synchronize()
    #     end = time.time()
        
    #     n_tokens = batch_size * seq_len * n_repeat
    #     elapsed  = end - start
    #     print(f'------> Batch Size : {batch_size}')
    #     print(f'total elapsed: {elapsed:.4f} s.')
    #     print(f'average elapsed: {elapsed/n_repeat:.4f} s.')
    #     print(f'average speed: {n_tokens * 3.29 / (elapsed*1e6):.3f} MB/s\n\n')