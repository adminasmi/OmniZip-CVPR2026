#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define TPB   256
#define TILE  16           // D、H 必须均为 16 的倍数

/* ------------------------------------------------------------------ */
/* forward kernel                                                     */
/* ------------------------------------------------------------------ */
__global__ void relu2_mlp_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    float*       __restrict__ out,
    int B, int D, int H)
{
    int b = blockIdx.y;
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D) return;

    extern __shared__ float sh[];
    float* x_row   = sh;           // size D
    float* w1_tile = sh + D;       // size TILE*D

    if (threadIdx.x < D)
        x_row[threadIdx.x] = x[b * D + threadIdx.x];
    __syncthreads();

    float y = 0.f;
    for (int j = 0; j < H; j += TILE) {
        #pragma unroll
        for (int t = 0; t < TILE; ++t)
            w1_tile[t * D + threadIdx.x] = w1[(j + t) * D + threadIdx.x];
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < D; ++i)
                dot += x_row[i] * w1_tile[t * D + i];
            float r = fmaxf(dot, 0.f);
            y += r * r * w2[d * H + j + t];
        }
        __syncthreads();
    }
    out[b * D + d] = y;
}

/* ------------------------------------------------------------------ */
/* element‑wise kernels                                               */
/* ------------------------------------------------------------------ */
__global__ void relu2_forward_ew(float* h, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = fmaxf(h[i], 0.f);
        h[i] = v * v;
    }
}

__global__ void relu2_backward_ew(float* grad_h, const float* h, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = h[i] > 0.f ? 2.f * h[i] : 0.f;
        grad_h[i] *= v;
    }
}

/* ------------------------------------------------------------------ */
/* host forward                                                       */
/* ------------------------------------------------------------------ */
torch::Tensor fused_relu2_forward(torch::Tensor x,
                                  torch::Tensor w1,
                                  torch::Tensor w2) {
    const int B = x.size(0);
    const int D = x.size(1);
    const int H = w1.size(0);

    auto y = torch::empty_like(x);

    dim3 block(TPB);
    dim3 grid((D + TPB - 1) / TPB, B);
    size_t smem = sizeof(float) * (D + TILE * D);

    const at::cuda::OptionalCUDAGuard guard(x.device());
    relu2_mlp_forward_kernel<<<grid, block, smem,
        at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w1.data_ptr<float>(), w2.data_ptr<float>(),
        y.data_ptr<float>(), B, D, H);

    return y;
}

/* ------------------------------------------------------------------ */
/* host backward (cuBLAS GEMM)                                        */
/* ------------------------------------------------------------------ */
std::vector<torch::Tensor> fused_relu2_backward(torch::Tensor x,
                                                torch::Tensor w1,
                                                torch::Tensor w2,
                                                torch::Tensor grad_y) {
    const int B = x.size(0);
    const int D = x.size(1);
    const int H = w1.size(0);

    auto h       = torch::empty({B, H}, x.options());   // x @ w1.T
    auto grad_h  = torch::empty_like(h);
    auto grad_x  = torch::empty_like(x);
    auto grad_w1 = torch::empty_like(w1);
    auto grad_w2 = torch::empty_like(w2);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.f, beta0 = 0.f;

    /* 1. h = x @ w1.T  -> (B,H) */
    at::cuda::blas::gemm<float>(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        B, H, D,
        &alpha,
        x.data_ptr<float>(),  B,
        w1.data_ptr<float>(), H,
        &beta0,
        h.data_ptr<float>(),  B);

    /* 2. in‑place ReLU²(h) */
    int N = h.numel();
    relu2_forward_ew<<<(N + 255) / 256, 256,
                       0, at::cuda::getCurrentCUDAStream()>>>(
        h.data_ptr<float>(), N);

    /* 3. grad_w2 = act.T @ grad_y   -> (H,D) */
    at::cuda::blas::gemm<float>(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, D, B,
        &alpha,
        h.data_ptr<float>(), H,
        grad_y.data_ptr<float>(), D,
        &beta0,
        grad_w2.data_ptr<float>(), H);

    /* 4. grad_h = grad_y @ w2       -> (B,H) */
    at::cuda::blas::gemm<float>(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        B, H, D,
        &alpha,
        grad_y.data_ptr<float>(), B,
        w2.data_ptr<float>(),     D,
        &beta0,
        grad_h.data_ptr<float>(), B);

    /* 5. grad_h *= 2*ReLU(h) */
    relu2_backward_ew<<<(N + 255) / 256, 256,
                        0, at::cuda::getCurrentCUDAStream()>>>(
        grad_h.data_ptr<float>(), h.data_ptr<float>(), N);

    /* 6. grad_w1 = grad_h.T @ x     -> (H,D) */
    at::cuda::blas::gemm<float>(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H, D, B,
        &alpha,
        grad_h.data_ptr<float>(), H,
        x.data_ptr<float>(),      D,
        &beta0,
        grad_w1.data_ptr<float>(), H);

    /* 7. grad_x  = grad_h @ w1      -> (B,D) */
    at::cuda::blas::gemm<float>(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        B, D, H,
        &alpha,
        grad_h.data_ptr<float>(), B,
        w1.data_ptr<float>(),     D,
        &beta0,
        grad_x.data_ptr<float>(), B);

    return {grad_x, grad_w1, grad_w2};
}

/* ------------------------------------------------------------------ */
/* pybind                                                             */
/* ------------------------------------------------------------------ */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fused_relu2_forward,  "ReLU2‑MLP forward (CUDA float32)");
    m.def("backward", &fused_relu2_backward, "ReLU2‑MLP backward (CUDA float32)");
}
