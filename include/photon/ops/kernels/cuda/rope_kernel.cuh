/**
 * @file rope_kernel.cuh
 * @brief CUDA RoPE (Rotary Position Embedding) kernel interface
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/rope_kernel.cu
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH

#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Precompute sin/cos cache for RoPE
 *
 * @param sin_cache Output sin cache [max_seq_len × head_size]
 * @param cos_cache Output cos cache [max_seq_len × head_size]
 * @param head_size Head dimension
 * @param max_seq_len Maximum sequence length
 * @param stream CUDA stream (optional)
 */
Result<void> rope_precompute_cuda(
    std::span<f32> sin_cache,
    std::span<f32> cos_cache,
    i32 head_size,
    i32 max_seq_len,
    cudaStream_t stream = nullptr);

/**
 * @brief Apply RoPE to Q and K tensors
 *
 * @param q Query tensor [dim]
 * @param k Key tensor [kv_dim]
 * @param sin_cache Sin cache [max_seq_len × head_size]
 * @param cos_cache Cos cache [max_seq_len × head_size]
 * @param pos Current position
 * @param dim Query dimension
 * @param kv_dim Key dimension
 * @param head_size Head size
 * @param stream CUDA stream (optional)
 */
Result<void> rope_cuda_launch(
    std::span<f32> q,
    std::span<f32> k,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH
