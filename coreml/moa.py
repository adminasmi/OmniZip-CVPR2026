import torch
import torch.nn as nn

class ReceptanceExpert(nn.Module):
    def __init__(self, n_embd, dim_att, high_rank, inference):
        super().__init__()
        self.n_embd = n_embd
        self.dim_att = dim_att
        self.high_rank = high_rank
        self.inference = inference
        
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance_A = nn.Linear(n_embd, high_rank, bias=False)
        self.receptance_B = nn.Linear(high_rank, dim_att, bias=False)
        self.init_weights()

    def init_weights(self):
        def _init_linear(linear, scale):
            linear.weight.data.uniform_(-scale, scale)
            
        for r in [self.receptance, self.receptance_A, self.receptance_B]:
            _init_linear(r, 0.5/(self.n_embd**0.5))
            
    def merge_branches(self):
        """
        W_fused = W + B @ A
        """
        W_main = self.receptance.weight.data      # [n_embd, n_embd]
        W_A = self.receptance_A.weight.data       # [high_rank, n_embd]
        W_B = self.receptance_B.weight.data       # [n_embd, high_rank]

        fused = W_main + torch.mm(W_B, W_A)

        self.receptance_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.receptance_reparam.weight.data = fused

        del self.receptance, self.receptance_A, self.receptance_B
        self.inference = True
            

    def forward(self, x):
        if self.inference:
            return self.receptance_reparam(x)
        else:
            return self.receptance(x) + self.receptance_B(self.receptance_A(x))
        
        
class ValueExpert(nn.Module):
    def __init__(self, n_embd, dim_att, high_rank, inference):
        super().__init__()
        self.n_embd = n_embd
        self.dim_att = dim_att
        self.high_rank = high_rank
        self.inference = inference
        
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.value_A = nn.Linear(n_embd, high_rank, bias=False)
        self.value_B = nn.Linear(high_rank, dim_att, bias=False)
        self.init_weights()

    def init_weights(self):
        def _init_linear(linear, scale):
            linear.weight.data.uniform_(-scale, scale)
            
        for v in [self.value, self.value_A, self.value_B]:
            _init_linear(v, 0.5/(self.n_embd**0.5))
            
    def merge_branches(self):
        """
        W_fused = W + B @ A
        """
        W_main = self.value.weight.data      # [n_embd, dim_att]
        W_A = self.value_A.weight.data       # [high_rank, n_embd]
        W_B = self.value_B.weight.data       # [dim_att, high_rank]

        fused = W_main + torch.mm(W_B, W_A)

        self.value_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.value_reparam.weight.data = fused

        del self.value, self.value_A, self.value_B
        self.inference = True
            

    def forward(self, x):
        if self.inference:
            return self.value_reparam(x)
        else:
            return self.value(x) + self.value_B(self.value_A(x))
        
        
class KeyExpert(nn.Module):
    def __init__(self, n_embd, dim_att, high_rank, inference):
        super().__init__()
        self.n_embd = n_embd
        self.dim_att = dim_att
        self.high_rank = high_rank
        self.inference = inference
        
        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.key_A = nn.Linear(n_embd, high_rank, bias=False)
        self.key_B = nn.Linear(high_rank, dim_att, bias=False)
        self.init_weights()

    def init_weights(self):
        def _init_linear(linear, scale):
            linear.weight.data.uniform_(-scale, scale)
            
        for k in [self.key, self.key_A, self.key_B]:
            _init_linear(k, 0.05/(self.n_embd**0.5))
            
    def merge_branches(self):
        """
        W_fused = W + B @ A
        """
        W_main = self.key.weight.data      # [n_embd, dim_att]
        W_A = self.key_A.weight.data       # [high_rank, n_embd]
        W_B = self.key_B.weight.data       # [dim_att, high_rank]

        fused = W_main + torch.mm(W_B, W_A)

        self.key_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key_reparam.weight.data = fused

        del self.key, self.key_A, self.key_B
        self.inference = True
            

    def forward(self, x):
        if self.inference:
            return self.key_reparam(x)
        else:
            return self.key(x) + self.key_B(self.key_A(x))