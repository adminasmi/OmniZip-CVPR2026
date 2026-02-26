import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_SIZE = 16
CHUNK_LEN = 16

def RUN_TORCH_RWKV7g(r: torch.Tensor,
                w: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                a: torch.Tensor,
                b: torch.Tensor,
) -> torch.Tensor:
    B, T, C = r.shape
    H = C // HEAD_SIZE
    N = HEAD_SIZE

    r = r.view(B, T, H, N)
    w = w.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    a = a.view(B, T, H, N)
    b = b.view(B, T, H, N)

    y = torch.empty(B, T, H, N, device=r.device, dtype=r.dtype)
    state = torch.zeros(B, H, N, device=r.device, dtype=r.dtype)

    outputs = []
    for t in range(T):
        rt = r[:, t]  # [B,H,N]
        wt = w[:, t].clamp(-60.0, 60.0)
        wt = torch.exp(-torch.exp(wt))  # [B,H,N]
        kt = k[:, t]
        at = a[:, t]
        bt = b[:, t]
        vt = v[:, t]

        sa = (at * state).sum(dim=-1)
        state = state * wt + sa.unsqueeze(-1) * bt + kt * vt

        # head_sum = Σ_j state_j * r_j  -> [B,H]
        head_sum = (state * rt).sum(dim=-1)
        outputs.append(head_sum.unsqueeze(-1).expand(-1, -1, N))
        # y[:, t] = head_sum.unsqueeze(-1).expand(-1, -1, N)
    
    y = torch.stack(outputs, dim=1).reshape(B, T, C)
    return y
    # return y.view(B, T, C)


def fuse_rkv(r, k, v, time_faaaa):
    ''' 
    `x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)` 
    '''
    B, T, C = r.shape
    H = C // HEAD_SIZE
    N = HEAD_SIZE

    r_t = r.view(B, T, H, N)
    k_t = k.view(B, T, H, N)
    v_t = v.view(B, T, H, N)

    tfa = time_faaaa.expand(B, T, H, N)
    coeff = (r_t * k_t * tfa).sum(dim=-1, keepdim=True)
    out_t = coeff * v_t

    return out_t.reshape(B, T, C)


class AttnBlk(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.inference = args.inference
        self.layer_id  = layer_id
        self.n_embd = args.n_embd
        self.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        with torch.no_grad():
            if args.n_layer > 1:
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            else:
                ratio_0_to_1 = 0
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x  = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k  = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v  = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -7 + 5 * (n / (self.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,self.dim_att))

            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))
            if layer_id != 0:
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,args.n_embd)+1.0)

            if self.inference:
                self.register_buffer('state_x', torch.zeros(1, 1, args.n_embd))
                self.register_buffer('state_decay', torch.zeros(1, 1, self.dim_att))
                self.register_buffer('state_v1', torch.zeros(1, 1, self.dim_att))
            else:
                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(args.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, self.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, self.dim_att, bias=False)

        self.output = nn.Linear(self.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att, eps=64e-5)

        if not self.inference:
            self._init_weights()


    def _init_weights(self):
        self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
        self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
        self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))

        self.output.weight.data.zero_()


    def forward(self, x, v1):
        B, T, C = x.size()
        H = self.n_head
        
        if self.inference:
            xx = self.state_x - x
            self.state_x = x.detach()
            
            current_decay = -F.softplus(-self.time_decay)
            w = current_decay + 0.9 * self.state_decay
            self.state_decay = w.detach()
        else:
            xx = self.time_shift(x) - x

        xrg = x + xx * self.time_maa_rg
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v

        r = self.receptance(xrg)
        w = -F.softplus(-(self.time_decay.expand(B, T, C))) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v1 = v
        else:
            v = v + (v1 - v) * torch.sigmoid(self.time_misc_v)
        g = torch.sigmoid(xrg)

        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = torch.sigmoid(self.time_aaaaa)

        ma = torch.sigmoid(self.time_misc_a)
        k = k * ma + k * a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k)
        k = k * torch.clamp(w * mk, max=0).exp()

        r,w,k,v,kk = [i.float().contiguous() for i in [r,w,k,v,kk]]

        x = RUN_TORCH_RWKV7g(
            r, w, k, v, -kk, (kk*a)
        )
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)
        
        residual = fuse_rkv(r, k, v, self.time_faaaa)
        x = x + residual
        # x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)
        
        x = self.output(x * g)
        return x, v1
    
    def _reset_states(self, batch_size=1):
        device = self.time_decay.device
        self.state_x  = torch.zeros(1, 1, self.n_embd).expand(batch_size, -1, -1).to(device)
        self.state_decay = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        self.state_v1 = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        
        
class AttnBlkHiRA(nn.Module):
    def __init__(self, args, layer_id, hira_factor=4):
        super().__init__()
        self.args = args
        self.inference = args.inference
        
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        self.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        with torch.no_grad():
            if args.n_layer > 1:
                ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            else:
                ratio_0_to_1 = 0
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -7 + 5 * (n / (self.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,self.dim_att))
            
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))
            if layer_id != 0:
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,args.n_embd)+1.0)

            if self.inference:
                self.register_buffer('state_x', torch.zeros(1, 1, args.n_embd))
                self.register_buffer('state_decay', torch.zeros(1, 1, self.dim_att))
                self.register_buffer('state_v1', torch.zeros(1, 1, self.dim_att))
            else:
                self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
            self.high_rank = int(args.n_embd * hira_factor)

            self.receptance = nn.Linear(args.n_embd, self.dim_att, bias=False)
            self.receptance_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
            self.receptance_B = nn.Linear(self.high_rank, self.dim_att, bias=False)
            
            self.key = nn.Linear(args.n_embd, self.dim_att, bias=False)
            self.key_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
            self.key_B = nn.Linear(self.high_rank, self.dim_att, bias=False)
            
            self.value = nn.Linear(args.n_embd, self.dim_att, bias=False)
            self.value_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
            self.value_B = nn.Linear(self.high_rank, self.dim_att, bias=False)
            
            self.output = nn.Linear(self.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, self.dim_att, eps=64e-5)
            
            if not self.inference:
                self._init_weights()
            
    def _init_weights(self):
        self.receptance.weight.data.uniform_(-0.5/(self.args.n_embd**0.5), 0.5/(self.args.n_embd**0.5))
        self.receptance_A.weight.data.uniform_(-0.5/(self.args.n_embd**0.5), 0.5/(self.args.n_embd**0.5))
        self.receptance_B.weight.data.uniform_(-0.05/(self.high_rank**0.5), 0.5/(self.high_rank**0.5))
        
        self.key.weight.data.uniform_(-0.05/(self.args.n_embd**0.5), 0.05/(self.args.n_embd**0.5))
        self.key_A.weight.data.uniform_(-0.05/(self.args.n_embd**0.5), 0.05/(self.args.n_embd**0.5))
        self.key_B.weight.data.uniform_(-0.05/(self.high_rank**0.5), 0.05/(self.high_rank**0.5))
        
        self.value.weight.data.uniform_(-0.5/(self.args.n_embd**0.5), 0.5/(self.args.n_embd**0.5))
        self.value_A.weight.data.uniform_(-0.5/(self.args.n_embd**0.5), 0.5/(self.args.n_embd**0.5))
        self.value_B.weight.data.uniform_(-0.5/(self.high_rank**0.5), 0.5/(self.high_rank**0.5))
        
        self.output.weight.data.zero_()
            
    
    def merge_branches(self):
        key_main = self.key.weight.data
        key_A  = self.key_A.weight.data
        key_B  = self.key_B.weight.data
        fused_key = key_main + torch.mm(key_B, key_A)
        
        self.key_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key_reparam.weight.data = fused_key
        
        value_main = self.value.weight.data
        value_A  = self.value_A.weight.data
        value_B  = self.value_B.weight.data
        fused_value = value_main + torch.mm(value_B, value_A)
        
        self.value_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.value_reparam.weight.data = fused_value
        
        receptance_main = self.receptance.weight.data
        receptance_A  = self.receptance_A.weight.data
        receptance_B  = self.receptance_B.weight.data
        fused_receptance = receptance_main + torch.mm(receptance_B, receptance_A)
        
        self.receptance_reparam = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.receptance_reparam.weight.data = fused_receptance
        
        del self.key, self.key_A, self.key_B
        del self.value, self.value_A, self.value_B
        del self.receptance, self.receptance_A, self.receptance_B
        
    def fused_forward(self, x, l0, lA, lB):
        return l0(x) + lB(lA(x))

    def forward(self, x, v1):
        B, T, C = x.size()
        H = self.n_head
        
        if self.inference:
            xx = self.state_x - x
            self.state_x = x.detach()
            
            current_decay = -F.softplus(-self.time_decay)
            w = current_decay + 0.9 * self.state_decay
            self.state_decay = w.detach()
        else:
            xx = self.time_shift(x) - x

        xrg = x + xx * self.time_maa_rg 
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v

        w = -F.softplus(-(self.time_decay.expand(B, T, C))) - 0.5
        if self.inference:
            k = self.key_reparam(xk)
            v = self.value_reparam(xv)
            r = self.receptance_reparam(xrg)
        else:
            r = self.fused_forward(xrg, self.receptance, self.receptance_A, self.receptance_B)
            k = self.fused_forward(xk,  self.key, self.key_A, self.key_B)
            v = self.fused_forward(xv,  self.value, self.value_A, self.value_B)
        
        if self.layer_id == 0:
            v1 = v
        else:
            v = v + (v1 - v) * torch.sigmoid(self.time_misc_v)
        g = torch.sigmoid(xrg)

        kk = F.normalize(k.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        a = torch.sigmoid(self.time_aaaaa)
        
        ma = torch.sigmoid(self.time_misc_a)
        k = k * ma + k * a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k)
        k = k * torch.clamp(w * mk, max=0).exp()

        r,w,k,v,kk = [i.float().contiguous() for i in [r,w,k,v,kk]]
        
        x = RUN_TORCH_RWKV7g(
            r, w, k, v, -kk, (kk*a)
        )
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        residual = fuse_rkv(r, k, v, self.time_faaaa)
        x = x + residual
        # x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)
        
        x = self.output(x * g)
        return x, v1
    
    def _reset_states(self, batch_size=1):
        device = self.time_decay.device
        self.state_x  = torch.zeros(1, 1, self.n_embd).expand(batch_size, -1, -1).to(device)
        self.state_decay = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        self.state_v1 = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        
        
