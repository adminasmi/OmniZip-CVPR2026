import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


@torch.jit.script
def cv_squared_jit(x: torch.Tensor):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    """
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor(0.0, device=x.device)
    return x.float().var() / (x.float().mean()**2 + eps)


@torch.jit.script
def gates_to_load_jit(gates: torch.Tensor):
    """Compute the true load per expert, given the gates.
    The load is the number of examples for which the corresponding gate is >0.
    """
    return (gates > 0).sum(dim=[0,1])  # Sum over batch and sequence dimensions


@torch.jit.script
def expert_balance_jit(counts: torch.Tensor, total_tokens: int):
    fractions = counts / (total_tokens + 1e-8)
    balance   = 1.0 - (fractions.std() / (fractions.mean() + 1e-8))
    return balance


@torch.jit.script
def prob_in_top_k_jit(
    clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor, k:int, mean: torch.Tensor, std: torch.Tensor
):
    """Helper function to NoisyTopKGating.
    Computes the probability that value is in top k, given different random noise.
    """
    # Reshape tensors to combine batch and sequence dimensions
    B, T, E = clean_values.shape
    clean_values_flat = clean_values.reshape(-1, E)
    noisy_values_flat = noisy_values.reshape(-1, E)
    noise_stddev_flat = noise_stddev.reshape(-1, E)
    noisy_top_values_flat = noisy_top_values.reshape(-1, noisy_top_values.size(-1))

    batch_size = clean_values_flat.size(0)  # B*T
    m = noisy_top_values_flat.size(1)
    top_values_flat = noisy_top_values_flat.flatten()

    # Calculate threshold positions
    threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + k
    threshold_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in).unsqueeze(1)
    is_in = torch.gt(noisy_values_flat, threshold_if_in)

    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out).unsqueeze(1)

    # Calculate probabilities using error function directly
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=clean_values.device))
    z_if_in = (clean_values_flat - threshold_if_in) / (noise_stddev_flat * sqrt_2)
    z_if_out = (clean_values_flat - threshold_if_out) / (noise_stddev_flat * sqrt_2)
    
    prob_if_in = 0.5 * (1 + torch.erf((z_if_in - mean) / std))  # Direct erf calculation
    prob_if_out = 0.5 * (1 + torch.erf((z_if_out - mean) / std))
    prob = torch.where(is_in, prob_if_in, prob_if_out)

    # Reshape back to original dimensions
    return prob.reshape(B, T, E)


class ExpertRouter(nn.Module):
    def __init__(self, n_embd, num_experts=3, k=1, noisy_gating=True, loss_coef=1e-2, logger=logging.getLogger('base')):
        super().__init__()
        self.n_embd  = n_embd
        self.num_experts = num_experts

        self.k = k      # top-k routing
        self.noisy_gating = noisy_gating
        self.loss_coef_aux = loss_coef
        self.loss_coef_z   = loss_coef * 0.1        # by default as 0.001
        
        self.logger = logger

        # Prev: Use Parameter instead of Linear for gating
        # self.  = nn.Parameter(torch.zeros(n_embd, num_experts), requires_grad=True)
        
        # Now: Use MLP for gating
        hidden_dim = max(16, n_embd // 2)
        self.router = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # Noise Projection
        self.w_noise = nn.Parameter(torch.zeros(n_embd, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        # nn.init.kaiming_uniform_(self.)
        
        for m in self.router:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.w_noise)

        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std',  torch.tensor([1.0]))


    def forward(self, x):
        """Forward pass for RWKV7 attention block routing
        Args:
            x: Tensor of shape [B, T, C]
        Returns:
            gates: Tensor [B, T, num_experts] routing probabilities
            aux_loss: auxiliary load balance loss
        """
        gates, load, clean_logits = self.noisy_top_k_gating(x, self.training)
        
        # Calculate load balancing loss
        importance = gates.sum(dim=[0,1])  # [num_experts]
        aux_loss = cv_squared_jit(importance) + cv_squared_jit(load)
        aux_loss = aux_loss * self.loss_coef_aux
        z_loss = (clean_logits ** 2).mean() * self.loss_coef_z

        return gates, aux_loss + z_loss
    
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating for RWKV7 attention blocks
        Args:
            x: [B, T, C] input tensor
        Returns:
            gates: [B, T, num_experts] routing probabilities
            load: [num_experts] expert load
        """
        B, T, C = x.shape

        # Project input to expert logits using matrix multiplication
        # Reshape x to [B*T, C] for batch matrix multiplication
        x_flat = x.reshape(-1, C)
        # clean_logits = x_flat @ self.w_gate  # [B*T, num_experts]
        
        clean_logits = self.router(x_flat)  # [B*T, num_experts]
        clean_logits = clean_logits.reshape(B, T, -1)  # Reshape back to [B, T, num_experts]

        if self.noisy_gating and train:
            # Add noise during training for exploration
            raw_noise_stddev = x_flat @ self.w_noise  # [B*T, num_experts]
            raw_noise_stddev = raw_noise_stddev.reshape(B, T, -1)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noise_stddev = None

        logits = self.softmax(logits)

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits  = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]

        # Normalize weights across top-k experts for each token
        top_k_gates = top_k_logits / (top_k_logits.sum(dim=-1, keepdim=True) + 1e-6)

        # Create gate matrix and scatter top-k weights
        gates = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        gates = (gates * top_k_gates.unsqueeze(-1)).sum(dim=-2)  # [B, T, num_experts]

        if self.noisy_gating and self.k < self.num_experts and train:
            # load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(dim=[0,1])
            load = prob_in_top_k_jit(clean_logits, noisy_logits, noise_stddev, top_logits, self.k, self.mean, self.std)
        else: 
            load = gates_to_load_jit(gates)

        return gates, load, clean_logits


class MoE(nn.Module):
    def __init__(self, args, expert_cls, expert_kwargs: dict, num_experts=3, k=1, stat_interval=1000):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  
        self.expert_cls = expert_cls

        self.router = ExpertRouter(n_embd=args.n_embd, num_experts=num_experts, k=k, noisy_gating=True)
        self.experts = nn.ModuleList([expert_cls(**expert_kwargs) for _ in range(num_experts)])
        
        # TODO: 之前设置的是每个模态有自己的 router
        # self.routers = nn.ModuleDict({
        #     modality: ExpertRouter(n_embd=args.n_embd, num_experts=num_experts, k=k, noisy_gating=True) for modality in args.modalities
        # })

        self.stat_interval = stat_interval       # record after each 100 batches
        self.stat_counter  = 0
        self._init_expert_stats()

    def forward(self, x, mode='train'):
        B, T, C = x.size()

        # router = self.routers[modality]
        gates, aux_loss = self.router(x)
        top_vals, top_idx = gates.topk(self.k, dim=-1)
        # weights = top_vals.softmax(dim=-1)
        weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-6)

        stat_interval = self.stat_interval if mode == 'train' else 10
        with torch.no_grad():
            if self.stat_counter % stat_interval == 0:
                expert_counts_batch = torch.zeros(self.num_experts, device=x.device)
                for expert_id in range(self.num_experts):
                    expert_counts_batch[expert_id] = (top_idx == expert_id).sum().float()
                self.expert_counts += expert_counts_batch
                self.token_counts += B * T

                balance_value = expert_balance_jit(expert_counts_batch, B * T * self.k)
                confidence_value = top_vals.max()

                self.expert_balance[0] = balance_value
                self.expert_confidence[0] = confidence_value

                if not torch._dynamo.is_compiling():
                    self.expert_usage_history.append({
                        'counts': expert_counts_batch,
                        'balance': balance_value.item(),
                        'confidence': confidence_value.item()
                    })
            self.stat_counter += 1

        output_combined = self.forward_parallel(x, weights=weights, top_idx=top_idx)
        return output_combined, aux_loss
    
    
    def forward_sequential(self, x, B, weights, top_idx):
        output_combined = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            for b in range(B):
                batch_mask = (top_idx[b] == expert_id).any(dim=-1)  # (T)
                token_indices = torch.where(batch_mask)[0]
                if len(token_indices) == 0:
                    continue

                x_tokens = x[b, token_indices]
                weights_selected = weights[b, token_indices]        # (num_tokens, k)
                pos_in_topk = (top_idx[b, token_indices] == expert_id).nonzero(as_tuple=True)[1]
                weights_selected = weights_selected.gather(1, pos_in_topk.unsqueeze(-1)).squeeze(-1)
                
                expert_output = expert(x_tokens.unsqueeze(0))[0, :]
                if len(token_indices) > 0:
                    output_combined[b].index_add_(0, token_indices, expert_output * weights_selected.unsqueeze(-1))
                    
        return output_combined
    
    def forward_parallel(self, x, weights, top_idx):
        """
        A faster parallel version that batches expert tokens across all sequences.
        """
        B, T, C = x.size()
        x = x.view(B*T, C)
        top_idx = top_idx.view(B*T, self.k)
        weights = weights.view(B*T, self.k)
        
        output_combined = torch.zeros_like(x)
        
        for expert_id, expert in enumerate(self.experts):
            token_mask = (top_idx == expert_id).any(dim=-1)  # (B*T)
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue
            
            x_tokens = x[token_indices]
            weights_selected = weights[token_indices]
            pos_in_topk = (top_idx[token_indices] == expert_id).nonzero(as_tuple=True)[1]
            weights_selected = weights_selected.gather(1, pos_in_topk.view(-1,1)).squeeze(1)       # (B*T)
            
            expert_output = expert(x_tokens.unsqueeze(0))[0]
            output_combined.index_add_(0, token_indices, expert_output * weights_selected.unsqueeze(-1))

        output_combined = output_combined.view(B, T, C)        
        return output_combined
    
    def _init_expert_stats(self):
        self.register_buffer('expert_counts', torch.zeros(self.num_experts))
        self.register_buffer('token_counts', torch.zeros(1))
        self.register_buffer('expert_balance', torch.zeros(1))
        self.register_buffer('expert_confidence', torch.zeros(1))
        self.expert_usage_history = []
        
    def reset_expert_stats(self):
        device = self.expert_counts.device
        self._init_expert_stats()
        for name, buf in self.named_buffers():
            setattr(self, name, buf.to(device))

    def get_expert_stats(self):
        stats = {
            'expert_counts': self.expert_counts.clone(),
            'token_counts': self.token_counts.clone(),
            'expert_balance': self.expert_balance.clone(),
            'expert_confidence': self.expert_confidence.clone(),
            'expert_usage_history': self.expert_usage_history.copy()
        }
        return stats
    

class OutputExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output = nn.Linear(args.n_embd, args.n_embd, bias=False)
        nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        return self.output(x)
    
    
def _log_expert_stats(writer, step, layer_stats, prefix):
            """
            Log expert statistics to TensorBoard
            
            Args:
                writer: TensorBoard SummaryWriter instance
                step: Current training step
                layer_stats: Dictionary of layer statistics
                prefix: Prefix for logging keys
            """
            expert_counts = layer_stats['expert_counts']
            token_counts  = layer_stats['token_counts']
            balance_value = layer_stats['expert_balance'].item() \
                if layer_stats['expert_balance'].numel() == 1 else layer_stats['expert_balance'][0].item()
            confidence_value = layer_stats['expert_confidence'].item() \
                if layer_stats['expert_confidence'].numel() == 1 else layer_stats['expert_confidence'][0].item()
            
            if token_counts > 0:    # Normalize usage
                normalized_counts = expert_counts / token_counts.item()
                for exp_idx, count in enumerate(normalized_counts):
                    writer.add_scalar(f'{prefix}/usage_expert_{exp_idx}', count.item(), step)
            writer.add_scalar(f'{prefix}/balance', balance_value, step)
            writer.add_scalar(f'{prefix}/confidence', confidence_value, step)
            
            
def _plot_expert_usage(history, output_dir, num_experts, layer_id, label_prefix):
    if not history:
        return
    
    steps, counts = range(len(history)), np.array([h['counts'].cpu().numpy() for h in history])
    metrics = {
        'usage': [(steps, counts[:,e], f'Expert {e}') for e in range(num_experts)],
        'balance': (steps, [h['balance'] for h in history], 'Balance Score'),
        'confidence': (steps, [h['confidence'] for h in history], 'Avg Confidence')
    }
    
    for name, data in metrics.items():
        plt.figure(figsize=(10, 6 if name == 'usage' else 4))
        if name == 'usage':
            for x, y, label in data: plt.plot(x, y, label=label)
            plt.legend()
        else:
            plt.plot(*data[:2])
        plt.title(f'Layer {layer_id} Expert {name.title()}')
        plt.xlabel('Step'); plt.ylabel(data[2])
        plt.ylabel(f'{name.upper()}')
        plt.savefig(f'{output_dir}/layer_{layer_id}_{label_prefix}_{name}.png')
        plt.close()