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


@MODULE_BUILD_FUNCS.register_with_name(module_name='rwkv7_hira_rmoa_moe')
def build_rwkv7_hira(args):
    from attnmoablks import AttnBlk_RMoA
    model = BaseRWKV_HIRA_MOA_MOE(args, attn_blk_cls=AttnBlk_RMoA, use_mlp_moe=True)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return model, criterion


if __name__ == '__main__':
    torch.cuda.set_device(6)
    
    @dataclass
    class ARGS:
        k: int = 2
        mlp_factor: int = 4
        num_moe_layers: int = 1
        num_experts: int = 4
        vocab_size : int = 16384
        n_layer : int = 2
        n_embd : int  = 96
        hira_factor: int = 4
        inference: bool  = False
    args = ARGS()
    print(args)
    
    batch_size = 2
    seq_len = 1024
    
    model, criterion = build_rwkv7_hira(args)
    model.count_params()
    model = model.cuda()

    text_inputs = torch.randint(0, args.vocab_size, (batch_size, seq_len)).cuda()
    target_tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len)).cuda()
    
    text_logits, aux_loss = model(text_inputs)
    print(f'Text logits: ({text_logits.shape}), Aux loss: {aux_loss.item():.4f}')
    
    # Calculate cross-entropy loss
    logits_flat = text_logits.view(-1, args.vocab_size)  # [batch_size*seq_len, vocab_size]
    targets_flat = target_tokens.view(-1)  # [batch_size*seq_len]
    
    # Calculate loss only on valid tokens (where mask is True)
    ce_loss = criterion(logits_flat, targets_flat)
    mean_ce_loss = ce_loss.mean()
    print(f'Cross-entropy loss: {mean_ce_loss.item():.4f}')
    
    # Combine with MoE auxiliary loss
    total_loss = mean_ce_loss + aux_loss
    print(f'Total loss: {total_loss.item():.4f} = CE loss ({mean_ce_loss.item():.4f}) + Aux loss ({aux_loss.item():.4f})')
    
    # Backward pass (if training)
    total_loss.backward()
    
    stats = model.get_expert_stats()
    print("Expert usage statistics:")
    for layer_key, layer_stats in stats.items():
        for moe_key, moe_stats in layer_stats.items():
            print(f"  {moe_key}:")
            print(f"    Total tokens: {moe_stats['token_counts'].item()}")
            print(f"    Expert balance: {moe_stats['expert_balance'].item():.4f}")
            print(f"    Expert counts: {[count.item() for count in moe_stats['expert_counts']]}")
            
    # === Optimizer update test ===
    print("\n=== Optimizer update test ===")
    class OPT_ARGS:
        lr = 1e-3
        betas = (0.9, 0.999)
        eps = 1e-8

    opt_args = OPT_ARGS()
    optimizer = model.configure_optimizers(opt_args)

    param_name, param = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
    param_before = param.detach().clone()

    print(f"Before step: {param_name}, mean={param_before.mean().item():.6f}, std={param_before.std().item():.6f}")

    optimizer.step()
    optimizer.zero_grad()

    param_after = getattr(model, param_name.split('.')[0]).state_dict()[param_name.split('.')[1]]
    delta = (param_after - param_before).abs().mean().item()

    print(f"After step: {param_name}, mean={param_after.mean().item():.6f}, std={param_after.std().item():.6f}")
    print(f"Mean |Δparam| = {delta:.6e}")

    if delta < 1e-10:
        print("Warning: Parameter update is too small — check gradient flow or optimizer config.")
    else:
        print("Optimizer update verified: parameters changed successfully.")

    # 检查梯度是否被清空
    grads_nonzero = [n for n, p in model.named_parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    print(f"Non-zero grads after optimizer.step(): {len(grads_nonzero)} parameters")