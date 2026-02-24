#include <torch/extension.h>
#include "ATen/ATen.h"

// ExpertRouter CUDA functions
void cuda_expert_router_forward(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    bool noisy_switch, bool train, double noise_epsilon
);

void cuda_expert_router_backward(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    float *dx, float *dw_gate, float *dw_noise,
    bool noisy_switch, bool train
);

// MoE CUDA functions
void cuda_moe_forward(
    int B, int T, int C, int E, int K,
    float *x, float *gates, float *v1,
    float *output_combined, float *output_v1,
    bool is_attn_expert
);

// ExpertRouter forward implementation
void expert_router_forward(
    int64_t B, int64_t T, int64_t C, int64_t E, int64_t K,
    torch::Tensor &x, torch::Tensor &w_gate, torch::Tensor &w_noise,
    torch::Tensor &gates, torch::Tensor &load, torch::Tensor &aux_loss,
    bool noisy_switch, bool train, double noise_epsilon
) {
    cuda_expert_router_forward(
        B, T, C, E, K,
        x.data_ptr<float>(), w_gate.data_ptr<float>(), w_noise.data_ptr<float>(),
        gates.data_ptr<float>(), load.data_ptr<float>(), aux_loss.data_ptr<float>(),
        noisy_switch, train, noise_epsilon
    );
}

// ExpertRouter backward implementation
void expert_router_backward(
    int64_t B, int64_t T, int64_t C, int64_t E, int64_t K,
    torch::Tensor &x, torch::Tensor &w_gate, torch::Tensor &w_noise,
    torch::Tensor &gates, torch::Tensor &load, torch::Tensor &aux_loss,
    torch::Tensor &dx, torch::Tensor &dw_gate, torch::Tensor &dw_noise,
    bool noisy_switch, bool train
) {
    cuda_expert_router_backward(
        B, T, C, E, K,
        x.data_ptr<float>(), w_gate.data_ptr<float>(), w_noise.data_ptr<float>(),
        gates.data_ptr<float>(), load.data_ptr<float>(), aux_loss.data_ptr<float>(),
        dx.data_ptr<float>(), dw_gate.data_ptr<float>(), dw_noise.data_ptr<float>(),
        noisy_switch, train
    );
}

// MoE forward implementation
void moe_forward(
    int64_t B, int64_t T, int64_t C, int64_t E, int64_t K,
    torch::Tensor &x, torch::Tensor &gates, torch::Tensor &v1,
    torch::Tensor &output_combined, torch::Tensor &output_v1,
    bool is_attn_expert
) {
    cuda_moe_forward(
        B, T, C, E, K,
        x.data_ptr<float>(), gates.data_ptr<float>(), v1.data_ptr<float>(),
        output_combined.data_ptr<float>(), output_v1.data_ptr<float>(),
        is_attn_expert
    );
}

// Register the operators
TORCH_LIBRARY(moe, m) {
    m.def("expert_router_forward(int B, int T, int C, int E, int K, Tensor x, Tensor w_gate, Tensor w_noise, Tensor(a!) gates, Tensor(b!) load, Tensor(c!) aux_loss, bool noisy_switch, bool train, float noise_epsilon) -> ()");
    m.def("expert_router_backward(int B, int T, int C, int E, int K, Tensor x, Tensor w_gate, Tensor w_noise, Tensor gates, Tensor load, Tensor aux_loss, Tensor(a!) dx, Tensor(b!) dw_gate, Tensor(c!) dw_noise, bool noisy_switch, bool train) -> ()");
    m.def("moe_forward(int B, int T, int C, int E, int K, Tensor x, Tensor gates, Tensor v1, Tensor(a!) output_combined, Tensor(b!) output_v1, bool is_attn_expert) -> ()");
}

TORCH_LIBRARY_IMPL(moe, CUDA, m) {
    m.impl("expert_router_forward", &expert_router_forward);
    m.impl("expert_router_backward", &expert_router_backward);
    m.impl("moe_forward", &moe_forward);
} 