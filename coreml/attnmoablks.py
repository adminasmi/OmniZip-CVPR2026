import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreml.attnblks import RUN_TORCH_RWKV7g, fuse_rkv
from coreml.moe import MoE
from coreml.moa import ReceptanceExpert, ValueExpert, KeyExpert

HEAD_SIZE = 16
CHUNK_LEN = 16

class BaseAttnBlk_MoA(nn.Module):
    '''
    Base AttnBlk + HiRA + Mixture of Attention
    '''
    def __init__(self, args, layer_id, shared_branches, moa_branches, expert_classes):
        super().__init__()
        self.args = args
        self.inference = args.inference
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd
        self.dim_att = args.dim_att

        self.num_experts = getattr(args, 'num_experts_moa', args.num_experts)
        self.top_k = getattr(args, 'k_moa', args.k)
        self.hira_factor = getattr(args, 'hira_factor', 4)

        self.head_size = HEAD_SIZE
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        
        # ===== time decay & mixing parameters =====
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
            
        if self.inference:
            self.register_buffer('state_x', torch.zeros(1, 1, args.n_embd))
            self.register_buffer('state_decay', torch.zeros(1, 1, self.dim_att))
            self.register_buffer('state_v1', torch.zeros(1, 1, self.dim_att))
        else:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.high_rank = int(self.n_embd * self.hira_factor)
        
        # ===== Shared and MoA Branches =====
        self.shared = shared_branches
        self.moa = moa_branches
        self.expert_classes = expert_classes
        self._build_branches()
        
        # ==== output & normalization =====
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if not args.inference:
            self.init_weights()
            
    def merge_branches(self):
        # ===== 1. merge shared branches =====
        for name in self.shared:
            l0 = getattr(self, name).weight.data
            lA = getattr(self, f"{name}_A").weight.data
            lB = getattr(self, f"{name}_B").weight.data
            fused = l0 + torch.mm(lB, lA)
            
            fused_linear = nn.Linear(self.n_embd, self.dim_att, bias=False)
            fused_linear.weight.data.copy_(fused)
            setattr(self, f"{name}_reparam", fused_linear)

            delattr(self, name)
            delattr(self, f"{name}_A")
            delattr(self, f"{name}_B")

        # ===== 2. merge all MoA expert branches =====
        for moa_name in self.moa:
            moa_module = getattr(self, f"moa_{moa_name}")
            if not hasattr(moa_module, "experts"):
                continue
            for expert in moa_module.experts:
                expert.merge_branches()

        self.inference = True
        
    
    def _build_branches(self):
        expert_kwargs = dict(n_embd=self.n_embd, dim_att=self.dim_att, high_rank=self.high_rank, inference=self.inference)
        for name in ['receptance', 'key', 'value']:
            if name in self.shared:
                self._make_linear_branch(name)
            elif name in self.moa:
                expert_cls = self.expert_classes[name]
                moa_module = MoE(
                    args=self.args,
                    expert_cls=expert_cls,
                    expert_kwargs=expert_kwargs,
                    num_experts=self.num_experts,
                    k=self.top_k,
                )
                setattr(self, f'moa_{name}', moa_module)
                
    def _make_linear_branch(self, name):
        setattr(self, f"{name.lower()}", nn.Linear(self.n_embd, self.dim_att, bias=False))
        setattr(self, f"{name.lower()}_A", nn.Linear(self.n_embd, self.high_rank, bias=False))
        setattr(self, f"{name.lower()}_B", nn.Linear(self.high_rank, self.dim_att, bias=False))
        
    def init_weights(self):
        def _init_linear(linear, scale):
            linear.weight.data.uniform_(-scale, scale)
        
        for name in ['receptance', 'key', 'value']:
            for suffix in ["", "_A", "_B"]:
                if hasattr(self, name + suffix):
                    scale = 0.05/(self.n_embd**0.5) if "key" in name else 0.5/(self.n_embd**0.5)
                    _init_linear(getattr(self, name + suffix), scale)
        self.output.weight.data.zero_()
        
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
        
        # Forward shared or MoA branches
        outputs, aux_losses = {}, {}
        for name in ['receptance', 'key', 'value']:
            xin = dict(receptance=xrg, key=xk, value=xv)[name]
            if name in self.shared:
                if self.inference:
                    outputs[name] = getattr(self, name + "_reparam")(xin)
                else:
                    outputs[name] = self.fused_forward(xin, getattr(self, name), getattr(self, name + "_A"), getattr(self, name + "_B"))
            elif name in self.moa:
                outputs[name], aux_losses[name] = getattr(self, f'moa_{name}')(xin)
        
        r, k, v = outputs.get('receptance'), outputs.get('key'), outputs.get('value')
        aux_loss = sum(aux_losses.values()) if len(aux_losses) > 0 else 0.0
        
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

        x = RUN_TORCH_RWKV7g(r, w, k, v, -kk, (kk*a))
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        # x = x + ((r.contiguous().view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.contiguous().view(B,T,H,-1)).view(B,T,C)
        residual = fuse_rkv(r, k, v, self.time_faaaa)
        x = x + residual
        x = self.output(x * g)

        return x, v1, aux_loss
    
    
    def _reset_states(self, batch_size=1):
        device = self.time_decay.device
        self.state_x  = torch.zeros(1, 1, self.n_embd).expand(batch_size, -1, -1).to(device)
        self.state_decay = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        self.state_v1 = torch.zeros(1, 1, self.dim_att).expand(batch_size, -1, -1).to(device)
        
        
def AttnBlk_RMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['key', 'value'],
        moa_branches=['receptance'],
        expert_classes={
            'receptance': ReceptanceExpert,
        }
    )

def AttnBlk_KMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['receptance', 'value'],
        moa_branches=['key'],
        expert_classes={
            'key': KeyExpert,
        }
    )
    
def AttnBlk_VMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['receptance', 'key'],
        moa_branches=['value'],
        expert_classes={
            'value': ValueExpert,
        }
    )
    
def AttnBlk_KVMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['receptance'],
        moa_branches=['key', 'value'],
        expert_classes={
            'key': KeyExpert,
            'value': ValueExpert,
        }
    )
    
def AttnBlk_RKMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['value'],
        moa_branches=['receptance', 'key'],
        expert_classes={
            'receptance': ReceptanceExpert,
            'key': KeyExpert,
        }
    )
    
def AttnBlk_RVMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=['key'],
        moa_branches=['receptance', 'value'],
        expert_classes={
            'receptance': ReceptanceExpert,
            'value': ValueExpert,
        }
    )
    
def AttnBlk_RKVMoA(args, layer_id):
    return BaseAttnBlk_MoA(
        args=args,
        layer_id=layer_id,
        shared_branches=[],
        moa_branches=['receptance', 'key', 'value'],
        expert_classes={
            'receptance': ReceptanceExpert,
            'key': KeyExpert,
            'value': ValueExpert,
        }
    )