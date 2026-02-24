########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from dataclasses import dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attnblks import AttnBlk
import torch
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam

from registry import MODULE_BUILD_FUNCS
from models.common import RMSNorm, MLP

                
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.attn = AttnBlk(args, layer_id=layer_id)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        
        self.mlp = torch.jit.script(MLP(n_embd=args.n_embd, factor=8))
        
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(self.ln1(x), v1)
        x = x + x1
        
        x2 = self.mlp(self.ln2(x))
        x = x + x2
        return x, v1
    
    
class RWKV7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rwkv = nn.ModuleDict(
            dict(
                wte = nn.Embedding(args.vocab_size, args.n_embd),
                h = nn.ModuleList([Block(args, layer_id) for layer_id in range(args.n_layer)]),
            )
        )
        self.rms_norm = RMSNorm(normalized_shape=(args.n_embd, )).cuda()
        self.lm_head  = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        
    def forward(self, idx, return_logits=True):
        x = self.rwkv.wte(idx)
        x = self.rms_norm(x)
        x0 = x
        v1 = None
        
        for block in self.rwkv.h:
            x, v1 = block(x, v1, x0)
            assert x.isfinite().all() and v1.isfinite().all(), '`x` contains NaN or Inf.'
        x = self.rms_norm(x)
        logits = self.lm_head(x).float()
        
        logits = logits if return_logits else None
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
    
    
    def count_params(self):
        excluded_layers = ['wte', 'lm_head']
        nparams = 0
        for name, param in self.named_parameters():
            if not any(excl in name for excl in excluded_layers):
                nparams += param.numel()            
        print(f'num of params: {nparams/1024}K')
    
        
# @MODULE_BUILD_FUNCS.register_with_name(module_name='rwkv7')
def build_rwkv7_hira(args):
    model = RWKV7(args)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return model, criterion
        
        
if __name__ == '__main__':
    print(MODULE_BUILD_FUNCS.module_dict)
    @dataclass
    class ARGS:
        vocab_size : int = 16384
        n_layer : int = 2
        n_head : int = 8    # head dim 128 suggested by @Grad62304977
        n_embd : int = 96
    args = ARGS()
    print(args)
    
    model, criterion = build_rwkv7_hira(args)
    model = model.cuda()
    
    torch.save(model.state_dict(), './rwkv7.pth')
    
    # excluded_layers = ['key_A', 'key_B', 'value_A', 'value_B', 'receptance_A', 'receptance_B', 'wte', 'lm_head']
    # excluded_layers = ['wte', 'lm_head']
    # nparams = 0
    
    # for name, param in model.named_parameters():
    #     if not any(excl in name for excl in excluded_layers):
    #         nparams += param.numel()            
    # print(f'num of params: {nparams/1e3}K')
    
    # nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)