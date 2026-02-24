import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
logger = logging.getLogger('base')

try:
    from deepspeed.ops.adam import FusedAdam
    FUSED_ADAM_AVAILABLE = True
except ImportError:
    FUSED_ADAM_AVAILABLE = False
    logger.warning("FusedAdam not available, falling back to standard Adam")

from models.common import RMSNorm, MLP
from attnblks import AttnBlkHiRA
from moe import MoE, _log_expert_stats, _plot_expert_usage


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
            num_experts=getattr(args, 'num_experts_moe', args.num_experts), 
            k=getattr(args, 'k_moe', args.k)
        )
        else:
            self.mlp = MLP(n_embd=args.n_embd, factor=8)
            
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        # x1, v1, aux_loss_moa = self.attn(self.ln1(x), v1)
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
        mode = 'train' if self.inference==False else 'inference'
        if self.use_mlp_moe:
            x2, aux_loss_moe = self.mlp_moe(self.ln2(x), mode=mode)
            x = x + x2
            aux_loss = aux_loss + aux_loss_moe
        else:
            x2 = self.mlp(self.ln2(x))
            x = x + x2
            
        return x, v1, aux_loss
        
    def reset_expert_stats(self):
        if self.use_mlp_moe:
            self.mlp_moe.reset_expert_stats()
            
        for name, module in self.attn.named_children():
            if isinstance(module, MoE):
                module.reset_expert_stats()
                
    def get_expert_stats(self):
        stats = {}
        if self.use_mlp_moe:
            stats['mlp_moe'] = self.mlp_moe.get_expert_stats()
        
        for name, module in self.attn.named_children():
            if isinstance(module, MoE):
                stats[name] = module.get_expert_stats()
        return stats
    
    def switch_to_inference(self):
        if hasattr(self.attn, 'merge_branches'):
            self.attn.merge_branches()
        self.inference = True                



class BaseRWKV_HIRA_MOA_MOE(nn.Module):
    def __init__(self, args, attn_blk_cls, use_mlp_moe=True):
        super().__init__()
        self.inference  = args.inference
        self.num_experts_moe = getattr(args, 'num_experts_moe', args.num_experts)
        self.num_experts_moa = getattr(args, 'num_experts_moa', args.num_experts)
        self.num_moe_layers = args.num_moe_layers if args.num_moe_layers >= 0 else args.n_layer
        if use_mlp_moe == False: self.num_moe_layers = 0
        self.num_moa_layers = args.num_moa_layers if args.num_moa_layers >= 0 else args.n_layer
        
        self.embedding = nn.Embedding(args.vocab_size, args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        
        self.rms_norm = RMSNorm(normalized_shape=(args.n_embd,))
        
        self.blocks = nn.ModuleList()
        for layer_id in range(args.n_layer):
            if layer_id < (args.n_layer - self.num_moe_layers):
                use_mlp_moe = False
            else:
                use_mlp_moe = True
                
            if layer_id < (args.n_layer - self.num_moa_layers):
                self.blocks.append(
                    BaseBlockMoAMoE(args, layer_id, attn_blk_cls=AttnBlkHiRA, use_mlp_moe=use_mlp_moe)
                )
            else:
                self.blocks.append(
                    BaseBlockMoAMoE(args, layer_id, attn_blk_cls=attn_blk_cls, use_mlp_moe=use_mlp_moe)
                )
        self.layer_aux_losses = {}
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.rms_norm(x)
        x0 = x
        v1 = None
        
        total_aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            outputs = block(x, v1, x0)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                x, v1, aux_loss = outputs
                self.layer_aux_losses[f'layer_{i}'] = aux_loss.item()
                total_aux_loss += aux_loss
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                x, v1 = outputs
            else:
                raise ValueError(f"Unexpected number of outputs ({len(outputs)}) from block {i}")
            assert x.isfinite().all() and v1.isfinite().all(), '`x` contains NaN or Inf.'
            torch.cuda.synchronize()
            
        x = self.rms_norm(x)
        logits = self.lm_head(x).float()
        return logits, total_aux_loss
    
    def reset_expert_stats(self):
        for block in self.blocks:
            if isinstance(block, BaseBlockMoAMoE):
                block.reset_expert_stats()
    
    def get_expert_stats(self, writer=None, step=None):
        """
        Get expert statistics and optionally log to TensorBoard
        
        Args:
            writer: TensorBoard SummaryWriter instance
            step: Current training step
        
        Returns:
            Dictionary of statistics
        """
            
        stats = {}
        
        # Collect stats from each layer
        for i, block in enumerate(self.blocks):
            if not isinstance(block, BaseBlockMoAMoE):
                continue
            layer_stats = block.get_expert_stats()
            stats[f'layer_{i}'] = layer_stats
            
            if writer is None or step is None:
                continue
            
            try:
                moe_stats = layer_stats.get('mlp_moe', None)
                if moe_stats:
                    _log_expert_stats(writer, step, moe_stats, f'layer_{i}/MLP-MoE')
                
                for name in layer_stats.keys():
                    if name.startswith('moa'):
                        moe_stats = layer_stats.get(name, None)
                        if moe_stats:
                            _log_expert_stats(writer, step, moe_stats, f'layer_{i}/Attn-MoA-{name.split("_")[-1].upper()}')
                
                if f'layer_{i}' in self.layer_aux_losses:
                    writer.add_scalar(f'layer_{i}/aux_loss', self.layer_aux_losses[f'layer_{i}'], step)
            except Exception as e:
                logger.warning(f"Failed to log expert stats for layer {i}: {e}")
        
        return stats
    
    def visualize_expert_usage(self, output_dir='expert_analysis'):
        """
        Generate visualizations of expert usage patterns
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, block in enumerate(filter(lambda b: isinstance(b, BaseBlockMoAMoE), self.blocks)):
            layer_stats = block.get_expert_stats()
            
            def save_usage_to_csv(history, label_prefix, layer_id):
                if not history:
                    return
                steps = list(range(len(history)))

                # extract numeric arrays safely
                expert_counts = np.array([h["counts"].cpu().numpy() for h in history])
                balance = [h.get("balance", np.nan) for h in history]
                confidence = [h.get("confidence", np.nan) for h in history]

                df = pd.DataFrame({
                    "step": steps,
                    "balance": balance,
                    "confidence": confidence
                })
                for e in range(expert_counts.shape[1]):
                    df[f"expert_{e}"] = expert_counts[:, e]

                csv_path = os.path.join(
                    output_dir,
                    f"layer_{layer_id}_{label_prefix.replace('-', '_')}_expert_usage.csv"
                )
                df.to_csv(csv_path, index=False)
            
            # === MLP-MoE ===
            mlp_stats = layer_stats.get("mlp_moe", None)
            if mlp_stats and "expert_usage_history" in mlp_stats:
                history = mlp_stats['expert_usage_history']
                _plot_expert_usage(
                    history, output_dir, self.num_experts_moe, layer_id=i, label_prefix='MLP-MoE'
                )
                save_usage_to_csv(history, 'MLP-MoE', i)
            
            # === Attn-MoA Receptance ===
            for name, module_stats in layer_stats.items():
                if name.startswith('moa') and "expert_usage_history" in module_stats:
                    history = module_stats["expert_usage_history"]
                    _plot_expert_usage(
                        history, output_dir, self.num_experts_moa, layer_id=i, label_prefix='Attn-MoA'
                    )
                    save_usage_to_csv(history, f'Attn-MoA_{name}', i)
    
            
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
            if FUSED_ADAM_AVAILABLE:
                optimizer = FusedAdam(optim_groups, lr=args.lr, betas=args.betas, eps=args.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
                logger.info("Using FusedAdam optimizer")
            else:
                raise ImportError("FusedAdam not available")
        except Exception as e:
            logger.warning(f"Failed to initialize FusedAdam: {e}. Falling back to standard Adam.")
            optimizer = torch.optim.Adam(optim_groups, lr=args.lr, betas=args.betas, eps=args.eps)

        return optimizer
    
    def switch_to_inference(self):
        for block in self.blocks:
            if hasattr(block, 'switch_to_inference'):
                block.switch_to_inference()
                
    def count_params(self):
        excluded_layers = ['key_A', 'key_B', 'value_A', 'value_B', 'receptance_A', 'receptance_B', 'embedding', 'lm_head']
        nparams = 0
        for name, param in self.named_parameters():
            if not any(excl in name for excl in excluded_layers):
                nparams += param.numel()            
        print(f'\n--->>> num of params: {nparams/1024}K <<<---')
        print('\n\n\n')
        
    def load_from_checkpoint(self, checkpoint_path, strict=False):
        if isinstance(checkpoint_path, str):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            state_dict = checkpoint_path
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        model_dict = self.state_dict()
        fixed_dict = {}
        
        for key, value in state_dict.items():
            if key in model_dict:
                if value.shape != model_dict[key].shape:
                    logger.info(f"Shape mismatch for {key}: checkpoint shape {value.shape}, model shape {model_dict[key].shape}")
                    
                    if len(value.shape) == 0 and len(model_dict[key].shape) > 0:
                        expanded_value = torch.full_like(model_dict[key], value.item())
                        fixed_dict[key] = expanded_value
                        logger.info(f"  Converted scalar to tensor for {key}")
                    elif len(value.shape) > 0 and len(model_dict[key].shape) == 0:
                        fixed_dict[key] = value.mean()
                        logger.info(f"  Converted tensor to scalar for {key}")
                    elif len(value.shape) == 1 and len(model_dict[key].shape) == 1:
                        target_size = model_dict[key].size(0)
                        src_size  = value.size(0)
                        new_value = torch.zeros_like(model_dict[key])
                        copy_size = min(target_size, src_size)
                        new_value[:copy_size] = value[:copy_size]
                        fixed_dict[key] = new_value
                        logger.info(f"  Resized tensor from {value.shape} to {model_dict[key].shape} for {key}")
                    else:
                        logger.info(f"  Cannot convert shapes for {key}, skipping")
                else:
                    fixed_dict[key] = value
            else:
                fixed_dict[key] = value
    
        return self.load_state_dict(fixed_dict, strict=strict)