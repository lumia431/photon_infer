/**
 * @file rope_kernel.cuh
 * @brief CUDA kernels for Rotary Position Embedding
 * @version 0.1.0
 *
 * RoPE applies 2D rotations to pairs of elements using precomputed sin/cos values.
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief CUDA kernel to precompute sin/cos cache
 *
 * Grid:  1 block
 * Block: head_size threads
 *
 * @param sin_cache Output sin values [max_seq_len × head_size]
 * @param cos_cache Output cos values [max_seq_len × head_size]
 * @param max_seq_len Maximum sequence length
 * @param head_size Dimension per attention head
 */
__global__ void compute_rope_cache_kernel(
    float* __restrict__ sin_cache,
    float* __restrict__ cos_cache,
    i32 max_seq_len,
    i32 head_size);

/**
 * @brief CUDA kernel for applying RoPE to Q and K
 *
 * Grid:  (dim + BLOCK_SIZE - 1) / BLOCK_SIZE blocks
 * Block: BLOCK_SIZE threads
 *
 * @param query Query tensor [dim] (modified in-place)
 * @param key Key tensor [kv_dim] (modified in-place)
 * @param sin_cache Precomputed sin values [max_seq_len × head_size]
 * @param cos_cache Precomputed cos values [max_seq_len × head_size]
 * @param pos Current position index
 * @param dim Query dimension
 * @param kv_dim Key/Value dimension
 * @param head_size Dimension per head
 */
template <i32 BLOCK_SIZE>
__global__ void rope_kernel(
    float* __restrict__ query,
    float* __restrict__ key,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size);

// ============================================================================
// Host-side Launch Functions
// ============================================================================

/**
 * @brief Launch RoPE cache computation kernel
 */
Result<void> compute_rope_cache_cuda_launch(
    std::span<f32> sin_cache,
    std::span<f32> cos_cache,
    i32 max_seq_len,
    i32 head_size,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch RoPE application kernel
 */
Result<void> rope_cuda_launch(
    std::span<f32> query,
    std::span<f32> key,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_ROPE_KERNEL_CUH
