import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda import RUN_CUDA_RWKV7g, HEAD_SIZE

class AttnBlk(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

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

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))

            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))
            if layer_id != 0:
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,args.n_embd)+1.0)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))
        self.key.weight.data.uniform_(-0.05/(self.n_embd**0.5), 0.05/(self.n_embd**0.5))
        self.value.weight.data.uniform_(-0.5/(self.n_embd**0.5), 0.5/(self.n_embd**0.5))

        self.output.weight.data.zero_()


    def forward(self, x, v1):
        B, T, C = x.size()
        H = self.n_head
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

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, (kk*a))
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v1


class AttnBlkHiRA(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args   = args
        self.inference = args.inference
        
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

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

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))
            
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))
            if layer_id != 0:
                self.time_misc_v = nn.Parameter(torch.zeros(1,1,args.n_embd)+1.0)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
        self.high_rank = int(args.n_embd * args.hira_factor)

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
        self.receptance_B = nn.Linear(self.high_rank, args.dim_att, bias=False)
        
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
        self.key_B = nn.Linear(self.high_rank, args.dim_att, bias=False)
        
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value_A = nn.Linear(args.n_embd, self.high_rank,  bias=False)
        self.value_B = nn.Linear(self.high_rank, args.dim_att, bias=False)
        
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)
        
        if not args.inference:
            self.init_weights()
            
            
    def init_weights(self):
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
            
    
    @torch.jit.ignore
    def merge_branches(self):
        # Merge shared K/V
        for name in ['key', 'value', 'receptance']:
            l0 = getattr(self, name).weight.data
            lA = getattr(self, name + "_A").weight.data
            lB = getattr(self, name + "_B").weight.data
            fused = l0 + torch.mm(lB, lA)
            setattr(self, name + "_reparam", nn.Linear(self.n_embd, self.args.dim_att, bias=False))
            getattr(self, name + "_reparam").weight.data = fused
        
        del self.key, self.key_A, self.key_B
        del self.value, self.value_A, self.value_B
        del self.receptance, self.receptance_A, self.receptance_B


    def forward(self, x, v1):
        B, T, C = x.size()
        H = self.n_head
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
            r = self.receptance(xrg)
            ra = self.receptance_A(xrg)
            rb = self.receptance_B(ra)
            r = r + rb
            
            k = self.key(xk)
            ka = self.key_A(xk)
            kb = self.key_B(ka)
            k = k + kb
            
            v = self.value(xv)
            va = self.value_A(xv)
            vb = self.value_B(va)
            v = v + vb
        
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
        
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, (kk*a))
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v1
    
        
    
    
