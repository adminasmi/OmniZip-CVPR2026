import torch
from torch.utils.cpp_extension import load
from torch._dynamo import disable

HEAD_SIZE = 16
CHUNK_LEN = 16
CUDA_FLAGS = [
    '-res-usage', '--use_fast_math', '--extra-device-vectorization', 
    '-O3', '-Xptxas -O3', '-DRWKV_DEBUG_PRINT=1',   # for running
    # '-DTORCH_USE_CUDA_DSA=1', '-DRWKV_DEBUG_PRINT=1', '-g', '-G', '-O0', '-Xptxas -O0',   # for debug 
]
DTYPE = torch.float
XTYPE = torch.float
T_MAX = 8192

# Load CUDA extensions
FAST_CUDA  = False

if FAST_CUDA:
    load(name='wkv7', sources=[
            '/home/zhaoy/OmniZip-CVPR2026/models/cuda/backstepping_f32.cpp', 
            f'/home/zhaoy/OmniZip-CVPR2026/models/cuda/backstepping_f32_{1 if HEAD_SIZE < 128 else 2}.cu'
        ], 
        is_python_module=False, 
        verbose=True, 
        extra_cuda_cflags=CUDA_FLAGS + [f'-D_C_={HEAD_SIZE}', f'-D_CHUNK_LEN_={CHUNK_LEN}',] ,
        extra_include_paths='/home/zhaoy/OmniZip-CVPR2026/models/cuda'
    )
    class WKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            
            w,q,k,v,z,b = [i.contiguous() for i in [w,q,k,v,z,b]]
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        
        @staticmethod
        def backward(ctx, dy):
            assert dy.dtype == torch.bfloat16
            dy = dy.contiguous()
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db
    
    def RUN_CUDA_RWKV7g(q,w,k,v,a,b,headsz=8):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//headsz,headsz) for i in [q,w,k,v,a,b]]
        return WKV7.apply(w,q,k,v,a,b).view(B,T,HC)
    
else:
    # DTYPE = torch.bfloat16
    DTYPE = torch.float
    XTYPE = torch.float
    T_MAX = 8192
    CHUNK_LEN = 16
    
    load(
        name='wkv7g',
        sources=[
            '/home/zhaoy/OmniZip-CVPR2026/models/cuda/wkv7g_op.cpp',
            '/home/zhaoy/OmniZip-CVPR2026/models/cuda/wkv7g_cuda.cu'
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=CUDA_FLAGS+[f'-D_N_={HEAD_SIZE}', f'-D_CHUNK_LEN_={CHUNK_LEN}', f'-D_T_={T_MAX}']
    )
    
    class WKV7g(torch.autograd.Function):
        
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                A = T // CHUNK_LEN
                assert HEAD_SIZE == C // H
                assert T % CHUNK_LEN == 0
                assert all(i.dtype == DTYPE for i in [r,w,k,v,a,b])
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format).uniform_(-100, 100)
                saa = torch.empty((B, T, H, N), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
                sss = torch.empty((B, H, A, N, N), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format).uniform_(-100, 100)
                
                # assert [i.isfinite().all() for i in [r,w,k,v,a,b,y,saa,sss]]
                r,w,k,v,a,b,y,saa,sss = [i.contiguous() for i in [r,w,k,v,a,b,y,saa,sss]]
                
                torch.ops.wkv7g.forward(B, T, C, H, r, w, k, v, a, b, y, saa, sss)
                ctx.save_for_backward(r, w, k, v, a, b, saa, sss)
                
                return y
        
        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                N = HEAD_SIZE
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                A = T // CHUNK_LEN
                assert gy.dtype == DTYPE
                gy = gy.contiguous()
                r, w, k, v, a, b, saa, sss = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                ga = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gb = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                zzz = torch.empty((B, H, A-1, N, N), device=gy.device, dtype=XTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                torch.ops.wkv7g.backward(B, T, C, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb)
                del saa
                del sss
                del zzz
                return (gr, gw, gk, gv, ga, gb)
    
    def RUN_CUDA_RWKV7g(r, w, k, v, a, b):
        return WKV7g.apply(r, w, k, v, a, b)
    
    
def is_compile_time():
    return torch._dynamo.is_compiling()


# Export constants and loaded extensions
__all__ = ['HEAD_SIZE', 'CHUNK_LEN', 'CUDA_FLAGS', 'DTYPE', 'XTYPE', 'T_MAX', 'WKV7g', 'RUN_CUDA_RWKV7g', 'is_compile_time'] 