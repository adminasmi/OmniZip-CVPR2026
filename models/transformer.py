import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from deepspeed.ops.adam import FusedAdam

from registry import MODULE_BUILD_FUNCS
    
    
class MultiHeadDotProductionAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        """Multi-head dot-product attention module."""
        super().__init__()
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert (
            n_embd % n_head == 0
        ), "Embedding dimension must be divisible by number of heads."
        
        self.qlinear = nn.Linear(n_embd, n_embd, bias=False)
        self.klinear = nn.Linear(n_embd, n_embd, bias=False)
        self.vlinear = nn.Linear(n_embd, n_embd, bias=False)
        self.olinear = nn.Linear(n_embd, n_embd, bias=False)    # output layer.
        
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Tensor of shape (B, T, E)
            key: Tensor of shape (B, S, E)
            value: Tensor of shape (B, S, E)
            mask: Tensor of shape (B, 1, T, S) or None
        
        Returns:
            Tensor of shape (B, T, E)
        """
        B, T, E = q.size()
        S = k.size(1)
        
        # Linear Projections (B, T, E)
        Q = self.qlinear(q)
        K = self.klinear(k)
        V = self.vlinear(v)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, H, T, D)
        K = K.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)    # (B, H, T, S)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)        # (B, H, T, S)
        attn_output  = torch.matmul(attn_weights, V)    # (B, H, T, D)
        
        attn_output  = attn_output.transpose(1, 2).contiguous().view(B, T, E)   # (B, T, E)
        
        output = self.olinear(attn_output)
        return output
    
    
def sinusoid_position_encoding(sequence_length: int, hidden_size: int, max_timescale: float = 1e4) -> torch.Tensor:
    """
    Creates sinusoidal positional encodings.
    
    Args:
        sequence_length: Length of the sequences.
        hidden_size: Dimension of the positional encodings.
        max_timescale: Maximum timescale for the frequencies.
    
    Returns:
        Tensor of shape (1, sequence_length, hidden_size)
    """
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)     # (T, 1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2).float() * (-np.log(max_timescale) / hidden_size)
    )  # (D/2)
    pe = torch.zeros(sequence_length, hidden_size)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, T, D)
    return pe  # Broadcasting over batch dimension


class TransformerDecoder(nn.Module):
    """Transformer Decoder model."""
    
    def __init__(self, args, logger=None):
        super().__init__()
        if logger is None:
            logger = logging.getLogger('base')
        
        # Compatibility with both args (from registry) and config (direct instantiation)
        if hasattr(args, 'vocab_size'):
            self.vocab_size = args.vocab_size
            self.n_embd = getattr(args, 'n_embd', getattr(args, 'n_embd', 64))
            self.n_layer = getattr(args, 'n_layer', getattr(args, 'n_layer', 4))
            self.n_head = getattr(args, 'n_head', 8)
            self.widening_factor = getattr(args, 'widening_factor', 4)
        else:
            self.vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 256
            self.n_embd = getattr(args, 'n_embd', 64)
            self.n_layer = getattr(args, 'n_layer', 4)
            self.n_head = getattr(args, 'n_head', 8)
            self.widening_factor = getattr(args, 'widening_factor', 4)
        
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.embedding_scale = np.sqrt(self.n_embd)
        self.pos_encoding = sinusoid_position_encoding(sequence_length=2048, hidden_size=self.n_embd)
        self.layers = nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    'self_attn': MultiHeadDotProductionAttention(n_head=self.n_head, n_embd=self.n_embd),
                    'layer_norm1': nn.LayerNorm(self.n_embd),
                    'fc1': nn.Linear(self.n_embd, self.n_embd * self.widening_factor),
                    'fc2': nn.Linear(self.n_embd * self.widening_factor, self.n_embd),
                    'layer_norm2': nn.LayerNorm(self.n_embd)
                }
            )
            for _ in range(self.n_layer)
        ])
        self.outlayer = nn.Linear(self.n_embd, self.vocab_size)
        self.outlayer.weight.data.zero_()
        
        excluded_layers = ['embedding', 'outlayer']
        nparams = 0
        for name, param in self.named_parameters():
            if not any(excl in name for excl in excluded_layers):
                nparams += param.numel()
        logger.info(f'num of params: {nparams/1024}K.')
                
        
    def forward(self, inputs: torch.Tensor, return_logits=True) -> torch.Tensor:
        """
        Forward pass of the Transformer Decoder.
        
        Args:
            inputs: Tensor of shape (B, T) containing input token indices.
            return_logits: If True, return logits; otherwise return None.
        
        Returns:
            Logits of shape (B, T, V) or None
        """
        # Embed the inputs and scale
        embeddings = self.embedding(inputs) * self.embedding_scale      # (B, T, E)
        
        # Add positional encodings
        embeddings = embeddings + self.pos_encoding[:, :embeddings.size(1), :].to(embeddings.device)    # (B, T, E)
        
        h = embeddings
        T = h.size(1)
        causal_mask = torch.tril(torch.ones((T, T), device=h.device)).unsqueeze(0).unsqueeze(0)      # (1, 1, T, T)
        
        for layer in self.layers:
            attn_output = layer['self_attn'](h, h, h, mask=causal_mask)     # (B, T, E)
            h = layer['layer_norm1'](h + attn_output)
            
            # feedforward
            ff_output = layer['fc1'](h)     # (B, T, E * widening_factor)
            ff_output = F.gelu(ff_output)
            ff_output = layer['fc2'](ff_output)     # (B, T, E)
            
            h = layer['layer_norm2'](h + ff_output)
        
        logits = self.outlayer(h).float()       # (B, T, V)
        logits = logits if return_logits else None
        return logits
    
    def configure_optimizers(self, args):
        """Configure optimizer like RWKV models."""
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
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
        """Count parameters like RWKV models."""
        excluded_layers = ['embedding', 'outlayer']
        nparams = 0
        for name, param in self.named_parameters():
            if not any(excl in name for excl in excluded_layers):
                nparams += param.numel()            
        print(f'--->>> num of params: {nparams/1024}K <<<---\n\n')


@MODULE_BUILD_FUNCS.register_with_name(module_name='transformer')
def build_transformer(args, **kwargs):
    """Build transformer model and criterion."""
    model = TransformerDecoder(args)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return model, criterion


if __name__ == '__main__':
    torch.cuda.set_device(2)
    print(MODULE_BUILD_FUNCS.module_dict)
    
    @dataclass
    class ARGS:
        vocab_size: int = 16384
        n_embd: int = 368
        n_layer: int = 2
        n_head: int = 8
        widening_factor: int = 4
        lr: float = 1e-4
        betas: tuple = (0.9, 0.999)
        eps: float = 1e-8
    
    args = ARGS()
    print(args)
    
    # Test with registry
    model, criterion = build_transformer(args)
    model.count_params()
    model = model.cuda()
    
    indices = torch.randint(0, args.vocab_size, (1, 16)).cuda()
    dummy_inputs = (indices,)
    
    print('\nProfile by fvcore.')
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_inputs)
    print(f'FLOPs: {flops.total() / (16 * 1024**2):.2f}M FLOPs.\n')
    print(f'MACs: {flops.total() / (2 * 16 * 1024**2):.2f}M MACs.\n\n')
        