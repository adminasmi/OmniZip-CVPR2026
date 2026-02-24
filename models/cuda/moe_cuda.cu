#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include "ATen/ATen.h"

// -----------------------------------------------------------------------------
// Debug macro to control printing of checks and logs
// -----------------------------------------------------------------------------
#ifndef RWKV_DEBUG_PRINT
#define RWKV_DEBUG_PRINT 0
#endif

#if RWKV_DEBUG_PRINT
#define RWKV_PRINT(...) printf(__VA_ARGS__)
#else
#define RWKV_PRINT(...) /* no-op */
#endif

// -----------------------------------------------------------------------------
// A helper macro to do CUDA-based indexing checks in debug mode
// -----------------------------------------------------------------------------
#if RWKV_DEBUG_PRINT
  #define RWKV_CHECK_BOUNDS(idx, limit, msg) do { \
      if ((idx) < 0 || (idx) >= (limit)) { \
          printf("[ERROR] %s out-of-bounds: idx=%d, limit=%d\n", (msg), (int)(idx), (int)(limit)); \
          return; \
      } \
  } while(0)
#else
  #define RWKV_CHECK_BOUNDS(idx, limit, msg) /* no-op */
#endif

// -----------------------------------------------------------------------------
// A helper to check float range (naive). If x is Inf or NaN or extremely large, print a warning.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void check_float_value(const float x, const char* msg, int idx) {
#if RWKV_DEBUG_PRINT
    if (isnan(x)) {
        printf("[WARNING] %s idx=%d is NaN\n", msg, idx);
    } else if (isinf(x)) {
        printf("[WARNING] %s idx=%d is Inf\n", msg, idx);
    } else if (fabsf(x) > 1.0e20f) {
        printf("[WARNING] %s idx=%d is too large: %e\n", msg, idx, x);
    }
#endif
}

// Add curand states for random number generation
__device__ curandState curand_states[1024]; // Assuming max 1024 threads

// Kernel to initialize curand states
__global__ void init_curand_states() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 1024) {
        curand_init(clock64(), tid, 0, &curand_states[tid]);
    }
}

// -----------------------------------------------------------------------------
// CUDA kernels for ExpertRouter
// -----------------------------------------------------------------------------

// Matrix multiplication kernel for expert routing
__global__ void matrix_multiply_kernel(
    int B, int T, int C, int E,
    const float* x, const float* w,
    float* output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < B*T && idy < E) {
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += x[idx * C + i] * w[i * E + idy];
        }
        output[idx * E + idy] = sum;
    }
}

// Softmax kernel
__global__ void softmax_kernel(
    int B, int T, int E,
    float* logits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < B*T) {
        // Find max for numerical stability
        float max_val = logits[idx * E];
        for (int i = 1; i < E; i++) {
            max_val = max(max_val, logits[idx * E + i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < E; i++) {
            logits[idx * E + i] = expf(logits[idx * E + i] - max_val);
            sum += logits[idx * E + i];
        }
        
        // Normalize
        for (int i = 0; i < E; i++) {
            logits[idx * E + i] /= sum;
        }
    }
}

// Top-k selection kernel
__global__ void top_k_kernel(
    int B, int T, int E, int K,
    const float* logits,
    float* top_values,
    int* top_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < B*T) {
        // Create a local copy of the logits for this token
        float local_logits[1024]; // Assuming E <= 1024
        int local_indices[1024];
        
        for (int i = 0; i < E; i++) {
            local_logits[i] = logits[idx * E + i];
            local_indices[i] = i;
        }
        
        // Simple bubble sort for top-k (not efficient but simple)
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < E - i - 1; j++) {
                if (local_logits[j] < local_logits[j + 1]) {
                    // Swap values
                    float temp_val = local_logits[j];
                    local_logits[j] = local_logits[j + 1];
                    local_logits[j + 1] = temp_val;
                    
                    // Swap indices
                    int temp_idx = local_indices[j];
                    local_indices[j] = local_indices[j + 1];
                    local_indices[j + 1] = temp_idx;
                }
            }
        }
        
        // Copy top-k values and indices
        for (int i = 0; i < K; i++) {
            top_values[idx * K + i] = local_logits[i];
            top_indices[idx * K + i] = local_indices[i];
        }
    }
}

// Create gates from top-k indices and values
__global__ void create_gates_kernel(
    int B, int T, int E, int K,
    const float* top_values,
    const int* top_indices,
    float* gates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < B*T && idy < E) {
        // Initialize gate to 0
        gates[idx * E + idy] = 0.0f;
        
        // Check if this expert is in the top-k
        for (int k = 0; k < K; k++) {
            if (top_indices[idx * K + k] == idy) {
                gates[idx * E + idy] = top_values[idx * K + k];
                break;
            }
        }
    }
}

// Calculate load for each expert
__global__ void calculate_load_kernel(
    int B, int T, int E,
    const float* gates,
    float* load
) {
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (expert_id < E) {
        float sum = 0.0f;
        for (int i = 0; i < B*T; i++) {
            if (gates[i * E + expert_id] > 0.0f) {
                sum += 1.0f;
            }
        }
        load[expert_id] = sum;
    }
}

// Calculate auxiliary loss
__global__ void calculate_aux_loss_kernel(
    int E,
    const float* load,
    float* aux_loss
) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < E; i++) {
        mean += load[i];
    }
    mean /= E;
    
    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < E; i++) {
        float diff = load[i] - mean;
        var += diff * diff;
    }
    var /= E;
    
    // Calculate coefficient of variation squared
    float eps = 1e-10f;
    *aux_loss = var / (mean * mean + eps);
}

// -----------------------------------------------------------------------------
// CUDA kernels for MoE
// -----------------------------------------------------------------------------

// Process tokens for each expert
__global__ void process_tokens_kernel(
    int B, int T, int C, int E, int K,
    const float* x,
    const float* gates,
    const float* v1,
    float* output_combined,
    float* output_v1,
    bool is_attn_expert
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < B && idy < T) {
        // Find which experts are selected for this token
        for (int e = 0; e < E; e++) {
            float gate = gates[idx * T * E + idy * E + e];
            if (gate > 0.0f) {
                // For simplicity, we're just copying the input to the output
                // In a real implementation, you would call the expert's forward function
                for (int c = 0; c < C; c++) {
                    output_combined[idx * T * C + idy * C + c] += gate * x[idx * T * C + idy * C + c];
                    
                    if (is_attn_expert && v1 != nullptr) {
                        output_v1[idx * T * C + idy * C + c] += gate * v1[idx * T * C + idy * C + c];
                    }
                }
            }
        }
    }
}

// CUDA kernel declarations
__global__ void expert_router_forward_kernel(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    bool noisy_switch, bool train, double noise_epsilon
);

// -----------------------------------------------------------------------------
// CUDA function implementations
// -----------------------------------------------------------------------------

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
    }
}

// ExpertRouter forward implementation
void cuda_expert_router_forward(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    bool noisy_switch, bool train, double noise_epsilon
) {
    // Initialize curand states
    dim3 init_grid(1, 1);
    dim3 init_block(1024);
    init_curand_states<<<init_grid, init_block>>>();
    cudaDeviceSynchronize();
    
    // Calculate grid and block dimensions
    dim3 grid(B, T);
    dim3 block(256);

    // Launch kernel
    expert_router_forward_kernel<<<grid, block>>>(
        B, T, C, E, K,
        x, w_gate, w_noise,
        gates, load, aux_loss,
        noisy_switch, train, noise_epsilon
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

// ExpertRouter backward implementation
void cuda_expert_router_backward(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    float *dx, float *dw_gate, float *dw_noise,
    bool noisy_switch, bool train
) {
    // Implementation of backward pass
    // (Implementation details omitted for brevity)
}

// MoE forward implementation
void cuda_moe_forward(
    int B, int T, int C, int E, int K,
    float *x, float *gates, float *v1,
    float *output_combined, float *output_v1,
    bool is_attn_expert
) {
    // Initialize output tensors
    cudaMemset(output_combined, 0, B * T * C * sizeof(float));
    if (output_v1 != nullptr) {
        cudaMemset(output_v1, 0, B * T * C * sizeof(float));
    }
    
    // Process tokens for each expert
    dim3 block(16, 16);
    dim3 grid((B + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    process_tokens_kernel<<<grid, block>>>(B, T, C, E, K, x, gates, v1, output_combined, output_v1, is_attn_expert);
}

__global__ void expert_router_forward_kernel(
    int B, int T, int C, int E, int K,
    float *x, float *w_gate, float *w_noise,
    float *gates, float *load, float *aux_loss,
    bool noisy_switch, bool train, double noise_epsilon
) {
    // Get thread indices
    int b = blockIdx.x;  // batch index
    int t = blockIdx.y;  // sequence index
    int tid = threadIdx.x;
    
    // Allocate shared memory for this batch and sequence
    extern __shared__ float shared_mem[];
    float *clean_logits = shared_mem;
    float *noisy_logits = &clean_logits[E];  // Only need E elements per thread block
    float *noise_stddev = &noisy_logits[E];  // Only need E elements per thread block
    
    // Compute clean logits
    if (tid < E) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += x[b * T * C + t * C + c] * w_gate[c * E + tid];
        }
        clean_logits[tid] = sum;
    }
    __syncthreads();
    
    // Add noise if needed
    if (noisy_switch && train) {
        if (tid < E) {
            float noise_sum = 0.0f;
            for (int c = 0; c < C; c++) {
                noise_sum += x[b * T * C + t * C + c] * w_noise[c * E + tid];
            }
            noise_stddev[tid] = noise_sum;
            
            // Generate random noise using Box-Muller transform
            float u1 = curand_uniform(&curand_states[tid]);
            float u2 = curand_uniform(&curand_states[tid]);
            float noise = (float)noise_epsilon * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            
            noisy_logits[tid] = clean_logits[tid] + noise * noise_stddev[tid];
        }
    } else {
        if (tid < E) {
            noisy_logits[tid] = clean_logits[tid];
        }
    }
    __syncthreads();
    
    // Apply softmax
    if (tid < E) {
        float max_val = -INFINITY;
        for (int e = 0; e < E; e++) {
            max_val = fmaxf(max_val, noisy_logits[e]);
        }
        
        float sum = 0.0f;
        for (int e = 0; e < E; e++) {
            noisy_logits[e] = expf(noisy_logits[e] - max_val);
            sum += noisy_logits[e];
        }
        
        for (int e = 0; e < E; e++) {
            noisy_logits[e] /= sum;
        }
    }
    __syncthreads();
    
    // Select top-k experts and create gates
    if (tid < E) {
        // Initialize gates to 0
        gates[b * T * E + t * E + tid] = 0.0f;
        
        // Find if this expert is in top-k
        float val = noisy_logits[tid];
        int rank = 0;
        for (int e = 0; e < E; e++) {
            if (noisy_logits[e] > val) {
                rank++;
            }
        }
        
        // If in top-k, set gate value
        if (rank < K) {
            gates[b * T * E + t * E + tid] = val;
        }
    }
    __syncthreads();
    
    // Calculate load
    if (tid < E) {
        float expert_load = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                expert_load += gates[b * T * E + t * E + tid];
            }
        }
        load[tid] = expert_load;
    }
    __syncthreads();
    
    // Calculate auxiliary loss (only in first thread)
    if (tid == 0) {
        float total_load = 0.0f;
        for (int e = 0; e < E; e++) {
            total_load += load[e];
        }
        float mean_load = total_load / E;
        
        // Calculate variance
        float var = 0.0f;
        for (int e = 0; e < E; e++) {
            float diff = load[e] - mean_load;
            var += diff * diff;
        }
        var /= E;
        
        // Calculate coefficient of variation squared
        float eps = 1e-10f;
        aux_loss[0] = var / (mean_load * mean_load + eps);
    }
} 