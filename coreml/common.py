import time
import torch
import torch.nn as nn

COUNT_PARAMS = True
GEN_COLEML   = True

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(*normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return self.weight * x * inv_rms
    

class MLP(nn.Module):
    def __init__(self, n_embd, factor=5):
        super().__init__()
        self.factor = factor
        self.c_fc    = nn.Linear(n_embd, factor * n_embd // 2, bias=False)
        self.c_proj  = nn.Linear(factor * n_embd // 2, n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        
    # @torch.compile(mode='max-autotune')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tr = x.matmul(self.c_fc.weight.T)
        x_tr = torch.relu(x_tr).square()
        return x_tr.matmul(self.c_proj.weight.T)

    # def forward(self, x):
    #     if self.factor > 0:
    #         x = self.c_fc(x)
    #         x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
    #     x = self.c_proj(x)
    #     return x

    
if __name__ == '__main__':
    rms_norm = RMSNorm(normalized_shape=30)
    
    x = torch.randn(10, 20, 30)
    x_norm = rms_norm(x)
    
    print(x_norm.shape)
