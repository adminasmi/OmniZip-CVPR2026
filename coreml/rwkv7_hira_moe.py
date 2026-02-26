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

from coreml.moe import MoE
from coreml.common import RMSNorm, MLP, GEN_COLEML
from coreml.rwkv7_hira import Block as NormalBlock

import os
import numpy as np
import coremltools as ct

class BaseBlockMoAMoE(nn.Module):
    def __init__(self, args, layer_id, attn_blk_cls, use_mlp_moe=True):
        super().__init__()
        self.layer_id = layer_id
        self.inference = args.inference
        
        self.attn = attn_blk_cls(args=args, layer_id=layer_id)
        
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        
        self.use_mlp_moe = use_mlp_moe
        if use_mlp_moe:
            self.mlp_moe = MoE(
            args=args,
            expert_cls=MLP,
            expert_kwargs={'n_embd': args.n_embd, 'factor': args.mlp_factor},
            num_experts=args.num_experts, k=args.k
        )
            
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn_outputs = self.attn(self.ln1(x), v1)
        aux_loss = torch.tensor(0.0, device=x.device)
        if isinstance(attn_outputs, tuple):
            if len(attn_outputs) == 3:
                x1, v1, aux_loss_moa = attn_outputs
                aux_loss = aux_loss_moa
            else:
                x1, v1 = attn_outputs  # Example when we don't have `aux_loss_moa`
        else:
            x1 = attn_outputs
            v1 = None
            
        x = x + x1
        if self.use_mlp_moe:
            x2, aux_loss_moe = self.mlp_moe(self.ln2(x))
            x = x + x2
            aux_loss = aux_loss + aux_loss_moe
        else:
            x2 = self.mlp(self.ln2(x))
            x = x + x2
            
        return x, v1, aux_loss
        
    
    def switch_to_inference(self):
        if hasattr(self.attn, 'merge_branches'):
            self.attn.merge_branches()
        self.inference = True   


class BaseRWKV_HIRA_MOA_MOE(nn.Module):
    def __init__(self, args, attn_blk_cls, use_mlp_moe=True):
        super().__init__()
        self.inference  = args.inference
        self.num_experts = args.num_experts
        self.num_moe_layers = args.num_moe_layers if args.num_moe_layers >= 0 else args.n_layer
        
        # self.embedding = nn.Embedding(args.vocab_size, args.n_embd)
        # self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        # self.lm_head.weight.data.zero_()
        
        self.rms_norm = RMSNorm(normalized_shape=(args.n_embd,))
        
        self.blocks = nn.ModuleList()
        for layer_id in range(args.n_layer - args.num_moe_layers):
            self.blocks.append(     
                NormalBlock(args, layer_id)
            )
        for layer_id in range(args.n_layer - args.num_moe_layers, args.n_layer):
            self.blocks.append(
                BaseBlockMoAMoE(args, layer_id, attn_blk_cls=attn_blk_cls, use_mlp_moe=use_mlp_moe)
            )
        self.layer_aux_losses = {}
        
    def forward(self, x):
        # x = self.embedding(x)
        x = self.rms_norm(x)
        x0 = x
        v1 = None
        
        if self.inference:
            for block in self.blocks:
                block.attn._reset_states(batch_size=x.shape[0])
        
        total_aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            if isinstance(block, BaseBlockMoAMoE):
                x, v1, aux_loss = block(x, v1, x0)
                self.layer_aux_losses[f'layer_{i}'] = aux_loss.item()
                total_aux_loss += aux_loss
            else:
                x, v1 = block(x, v1, x0)
            assert x.isfinite().all() and v1.isfinite().all(), '`x` contains NaN or Inf.'
            torch.cuda.synchronize()
            
        x = self.rms_norm(x)
        # logits = self.lm_head(x).float()
        # return logits, total_aux_loss
        return x
    
    def switch_to_inference(self):
        for block in self.blocks:
            if hasattr(block, 'switch_to_inference'):
                block.switch_to_inference()
    
    
def gen_coreml(space, mode='cpu_only'):
    for s in space:
        size = s.pop('size')
        batch_size = s.pop('batch_size')
        
        args = ARGS(
            # k=2, mlp_factor=4, num_experts=3, num_moe_layers=1, vocab_sizes={'text': 16384, 'image': 256}, hira_factors={'text':4, 'image':2}, inference=True, **s
            inference=True, **s
        )
        from attnblks import AttnBlkHiRA
        model = BaseRWKV_HIRA_MOA_MOE(args, attn_blk_cls=AttnBlkHiRA, use_mlp_moe=True)
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
            
            save_dir = f'/home/zhaoy/OmniComp-MoA-MoE-cvpr2026/coreml/mlpackage/{mode}/rwkv7-hira-moe/rwkv7-hira-moe-{size}'
            os.makedirs(save_dir, exist_ok=True)
            converted_model.save(os.path.join(save_dir, coreml_model_name))
        
        
if __name__ == '__main__':
    torch.cuda.set_device(1)
    
    @dataclass
    class ARGS:
        k: int = 2
        mlp_factor: int = 4
        num_moe_layers: int = 2
        num_experts: int = 4
        vocab_size : int = 16384
        n_layer : int = 2
        n_embd : int  = 96
        hira_factor: int = 4
        inference: bool  = False
        
    from thop import profile    
    from coreml.attnmoablks import AttnBlk_VMoA, AttnBlk_KVMoA, AttnBlk_RKVMoA
    
    for batch_size in [1, 16, 128, 512, 1024]:
        space = [
            {'n_embd': 320,   'n_layer': 2, 'size': 's',   'batch_size': batch_size},
        ]
            
        if GEN_COLEML:
            gen_coreml(space=space, mode='all')