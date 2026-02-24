import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_embd, factor=8):
        super().__init__()
        self.factor = factor
        self.c_fc    = nn.Linear(n_embd, factor * n_embd // 2, bias=False)
        self.c_proj  = nn.Linear(factor * n_embd // 2, n_embd, bias=False)
        
        nn.init.kaiming_uniform_(self.c_fc.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.c_proj.weight)
        
    # @torch.compile(mode='max-autotune')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tr = x.matmul(self.c_fc.weight.T)
        x_tr = torch.relu(x_tr).square()
        return x_tr.matmul(self.c_proj.weight.T)
    

def load_from_image_checkpoint(model, checkpoint_path, strict=False):
    if isinstance(checkpoint_path, str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    else:
        state_dict = checkpoint_path

    if 'model' in state_dict:
        state_dict = state_dict['model']

    model_dict = model.state_dict()
    fixed_dict = {}
    loaded_keys = set()  # Track which keys were loaded from the checkpoint

    for key, value in state_dict.items():
        new_key = None

        # -----------------------
        # word embedding & head
        if key.startswith('rwkv.wte.weight'):
            new_key = f'embeddings.image.weight'
        elif key.startswith('lm_head.weight'):
            new_key = f'lm_heads.image.weight'

        # -----------------------
        # attention: modality-specific projection layers
        elif '.attn.' in key and any(x in key for x in ['key', 'value', 'receptance']):
            layer_id = key.split('.')[2]
            suffix = '.'.join(key.split('.')[-2:])
            new_key = f'blocks.{layer_id}.attn.modality_specifics.image.{suffix}'

        # -----------------------
        # attention: time_*
        elif '.attn.' in key and any(t in key for t in ['time_maa_', 'time_decay', 'time_faaaa', 'time_aaaaa', 'time_misc_']):
            layer_id = key.split('.')[2]
            suffix = '.'.join(key.split('.')[-1:])
            new_key = f'blocks.{layer_id}.attn.{suffix}.image'

        # -----------------------
        # everything else (shared structure)
        elif key in model_dict:
            new_key = key

        if new_key in model_dict and model_dict[new_key].shape == value.shape:
            fixed_dict[new_key] = value
            loaded_keys.add(new_key)  # Add to the set of loaded keys
            # print(f"[+] Mapped: {key} → {new_key}")
        else:
            pass
            # print(f"[ ] Skipped: {key} (no match or shape mismatch)")

    print(f"[loader] Transferred {len(fixed_dict)} parameters from checkpoint")
    missing_keys, unexpected_keys = model.load_state_dict(fixed_dict, strict=strict)
    print(f"[loader] Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # Freeze only the parameters that were loaded from the checkpoint
    frozen_params = 0
    for name, param in model.named_parameters():
        # Check if parameter belongs to an attention block and was loaded from the checkpoint
        if '.attn.' in name and 'output' not in name and 'lm_heads' not in name and 'embeddings' not in name:
            # Check if this parameter was loaded from the checkpoint
            if name in loaded_keys:
                param.requires_grad = False
                frozen_params += 1
    
    print(f"[loader] Frozen {frozen_params} attention block parameters loaded from checkpoint")
    
    return missing_keys, unexpected_keys


def test_speed(model, vocab_size, batch_sizes=[1,16,128,512,1024], seq_len=16):
    import time
    print('Benchmarking inference speed...\n')
    warmup = 5
    test_runs = 20
    
    for batch_size in batch_sizes:
        print(f'\n*======== Batch size: {batch_size}')
        indices = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        dummy_inputs = (indices,)

        # Warm-up 阶段
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*dummy_inputs)
            torch.cuda.synchronize()

        # 正式计时
        start = time.time()
        with torch.no_grad():
            for _ in range(test_runs):
                _ = model(*dummy_inputs)
            torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / test_runs  # 单次推理时间
        ntokens = batch_size * seq_len
        speed  = ntokens * 3.29 / avg_time

        print(f"Average inference time: {avg_time * 1000 / ntokens:.3f} ms/token ({ntokens} tokens).")
        print(f"Speed: {speed/1000:.3f} KB/s\n\n")
        
        
def cal_macs(model, vocab_size, batch_size=1, seq_len=16):
    indices = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    dummy_inputs = (indices,)
    
    print('\nProfile by fvcore.')
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_inputs)
    flops = flops.total() / (batch_size * seq_len)
    macs = flops / 2
    
    print(f'FLOPs: {flops / (1024**2):.3f}M FLOPs.\n')
    print(f'MACs:  {macs / (1024**2):.3f}M MACs.\n\n')


    
if __name__ == '__main__':
    rms_norm = RMSNorm(normalized_shape=30)
    
    x = torch.randn(10, 20, 30)
    x_norm = rms_norm(x)
    
    print(x_norm.shape)
