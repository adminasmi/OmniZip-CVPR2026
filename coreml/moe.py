import torch
import torch.nn as nn

import logging
import torch.nn.functional as F


@torch.jit.script
def cv_squared_jit(x: torch.Tensor):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    """
    eps = 1e-10
    # if x.shape[0] == 1:
    #     return torch.tensor(0.0, device=x.device)
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
        self.loss_coef = loss_coef

        self.logger = logger

        # Use Parameter instead of Linear for gating
        self.w_gate  = nn.Parameter(torch.zeros(n_embd, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(n_embd, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

        # Initialize weights
        nn.init.kaiming_uniform_(self.w_gate)
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
        gates, load = self.noisy_top_k_gating(x, self.training)
        
        # Calculate load balancing loss
        importance = gates.sum(dim=[0,1])  # [num_experts]
        aux_loss = cv_squared_jit(importance) + cv_squared_jit(load)
        aux_loss = aux_loss * self.loss_coef

        return gates, aux_loss
    
    
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
        clean_logits = x_flat @ self.w_gate  # [B*T, num_experts]
        clean_logits = clean_logits.reshape(B, T, -1)  # Reshape back to [B, T, num_experts]

        logits = clean_logits
        logits = self.softmax(logits)

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits  = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]

        # Normalize weights across top-k experts for each token
        top_k_gates = top_k_logits / (top_k_logits.sum(dim=-1, keepdim=True) + 1e-6)

        # Create gate matrix and scatter top-k weights
        gates = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        gates = (gates * top_k_gates.unsqueeze(-1)).sum(dim=-2)  # [B, T, num_experts]

        load = gates_to_load_jit(gates)

        return gates, load


class MoE(nn.Module):
    def __init__(self, args, expert_cls, expert_kwargs: dict, num_experts=3, k=1, stat_interval=1000):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.expert_cls = expert_cls

        # self.router = ExpertRouter(n_embd=n_embd, num_experts=num_experts, k=k, noisy_gating=True)
        self.experts = nn.ModuleList([expert_cls(**expert_kwargs) for _ in range(num_experts)])
        self.router = ExpertRouter(n_embd=args.n_embd, num_experts=num_experts, k=k, noisy_gating=True)

        self.stat_interval = stat_interval       # record after each 100 batches
        self.stat_counter  = 0
        self._init_expert_stats()
        
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


    def forward(self, x):
        B, T, C = x.size()

        gates, aux_loss = self.router(x)
        top_vals, top_idx = gates.topk(self.k, dim=-1)
        # weights = top_vals.softmax(dim=-1)
        weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-6)

        output_combined = self.forward_parallel(x, weights=weights, top_idx=top_idx)
        return output_combined, aux_loss
    
    def forward_parallel(self, x, weights, top_idx):
        B, T, C = x.size()
        K = self.k
        E = self.num_experts
        BT = B * T
        
        x_flat       = x.view(BT, C)
        top_idx_flat = top_idx.view(BT, K)  # (BT, k)
        w_flat       = weights.view(BT, K)   # (BT, k)

        expert_outs = torch.stack([expert(x_flat) for expert in self.experts], dim=0)
        expert_outs = expert_outs.permute(1, 0, 2)
        
        row_idx = torch.arange(BT, device=x.device).unsqueeze(1).expand(-1, K)
        indices = torch.stack([row_idx.reshape(-1), top_idx_flat.reshape(-1)], dim=0)
        
        weight_matrix = torch.zeros(BT, E, device=x.device, dtype=w_flat.dtype)
        weight_matrix = torch.bincount(
            indices[0] * E + indices[1],
            weights=w_flat.reshape(-1), 
            minlength=BT*E
        ).view(BT, E)
        
        out_flat = (weight_matrix.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out_flat.view(B, T, C)
    
    
    def forward_parallel(self, x, weights, top_idx):
        """
        x: (B, T, C)
        weights: (B, T, k)
        top_idx: (B, T, k)
        """
        B, T, C = x.size()
        K = self.k
        BT = B * T

        x_flat       = x.view(BT, C)                 # (BT, C)
        top_idx_flat = top_idx.view(BT, K)           # (BT, k)
        w_flat       = weights.view(BT, K)           # (BT, k)

        expert_outs = torch.stack([expert(x_flat) for expert in self.experts], dim=0)  # (E, BT, C)
        expert_outs = expert_outs.permute(1, 0, 2)                                   # (BT, E, C)

        out_flat = torch.zeros(BT, C, device=x.device, dtype=x.dtype)
        batch_idx = torch.arange(BT, device=x.device)
        for i in range(K):
            idx_i = top_idx_flat[:, i]        # (BT,)
            w_i   = w_flat[:, i].unsqueeze(1) # (BT, 1)
            expert_i_out = expert_outs[batch_idx, idx_i, :]
            out_flat += expert_i_out * w_i    # (BT, C)

        return out_flat.view(B, T, C)

    

class OutputExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output = nn.Linear(args.n_embd, args.n_embd, bias=False)
        nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        return self.output(x)