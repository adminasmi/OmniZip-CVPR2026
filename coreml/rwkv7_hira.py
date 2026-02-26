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
from thop import profile, clever_format

from coreml.common import RMSNorm, MLP, GEN_COLEML
from coreml.attnblks import AttnBlkHiRA

import os
import numpy as np
import coremltools as ct

    
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
    def __init__(self, args):
        super().__init__()
        self.inference = args.inference
        self.rwkv = nn.ModuleDict(
            dict(
                wte = nn.Embedding(args.vocab_size, args.n_embd),
                h = nn.ModuleList([Block(args, layer_id) for layer_id in range(args.n_layer)]),
            )
        )
        self.rms_norm = RMSNorm(normalized_shape=(args.n_embd, ))
        self.lm_head  = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        
    def forward(self, x):
        # x = self.rwkv.wte(idx)
        x = self.rms_norm(x)
        x0 = x
        v1 = None
        
        if self.inference:
            for block in self.rwkv.h:
                block.attn._reset_states(batch_size=x.shape[0])
        
        for block in self.rwkv.h:
            x, v1 = block(x, v1, x0)
        x = self.rms_norm(x)
        logits = self.lm_head(x).float()
        return logits
    
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

        
        
def gen_coreml(space, mode='cpu_only'):
    for s in space:
        size = s.pop('size')
        batch_size = s.pop('batch_size')
        
        args = ARGS(
            # k=2, mlp_factor=4, num_experts=3, num_moe_layers=1, vocab_sizes={'text': 16384, 'image': 256}, hira_factors={'text':4, 'image':2}, inference=True, **s
            inference=True, **s
        )
        model = RWKV7_HIRA(args)
        model.switch_to_inference()
        model.eval()

        with torch.no_grad():
            inputs_indices = torch.zeros([batch_size, 1, args.n_embd])
            dummy_inputs = (inputs_indices,)
            # get flops and parameters 
            flops, _ = profile(model, inputs=dummy_inputs)
            flops, _ = clever_format([flops, _])
            print(flops, _)
            
        traced_model = torch.jit.trace(model, dummy_inputs)
        converted_model = ct.convert(
            traced_model,
            # ct.RangeDim(1, 64) means Range for the sequence dimension to be between [1, 64],
            inputs=[
                ct.TensorType(name="input1", shape=inputs_indices.shape, dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT32 if mode == 'cpu_only' else ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.CPU_ONLY if mode == 'cpu_only' else ct.ComputeUnit.ALL,
        )
        coreml_model_name = f'rwkv7-hira-{size}_bs{batch_size}_{s["n_embd"]}e_{s["n_layer"]}l.mlpackage'
        
        save_dir = f'/home/zhaoy/OmniComp-MoA-MoE-cvpr2026/coreml/mlpackage/{mode}/rwkv7-hira/rwkv7-hira-{size}'
        os.makedirs(save_dir, exist_ok=True)
        converted_model.save(os.path.join(save_dir, coreml_model_name))
        
        
if __name__ == '__main__':
    @dataclass
    class ARGS:
        vocab_size : int = 16384
        n_layer : int = 1
        n_embd : int = 96
        inference : bool = True
        
    from thop import profile    
    
    for batch_size in [1, 16, 128, 512, 1024]:
        space = [
            # {'n_embd': 96,    'n_layer': 2, 'size': 'xs',  'batch_size': batch_size},
            {'n_embd': 320,   'n_layer': 2, 'size': 's',  'batch_size': batch_size},
            # {'n_embd': 912,   'n_layer': 2, 'size': 'm',  'batch_size': batch_size},
            # {'n_embd': 1488,  'n_layer': 3, 'size': 'l',   'batch_size': batch_size},
        ]
            
        if GEN_COLEML:
            gen_coreml(space=space, mode='all')