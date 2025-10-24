/**
 * @file mha_kernel.cuh
 * @brief CUDA kernels for Multi-Head Attention
 * @version 0.1.0
 *
 * Optimized MHA with:
 * - Float4 vectorized Q·K computation
 * - CUB-based softmax
 * - Efficient attention weighted sum
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_MHA_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_MHA_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * @brief Device function for in-place softmax
 *
 * Applies softmax to array x[0:size] in-place using CUB reduction.
 * Uses shared memory for max and sum reduction.
 *
 * @param x Input/output array (modified in-place)
 * @param size Number of elements
 */
__device__ void softmax_inplace(float* __restrict__ x, i32 size);

/**
 * @brief CUDA kernel for multi-head attention
 *
 * Computes attention for all heads in parallel:
 * 1. scores[t] = (Q[h] · K[h][t]) / sqrt(head_size)  for t in [0, pos]
 * 2. attn[h] = softmax(scores[h])
 * 3. output[h] = sum(attn[h][t] * V[h][t])
 *
 * Grid:  head_num blocks (one block per head)
 * Block: BLOCK_SIZE threads
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param query Query tensor [dim = head_num * head_size]
 * @param key_cache Key cache [layer_num × seq_len × kv_dim]
 * @param value_cache Value cache [layer_num × seq_len × kv_dim]
 * @param output Output tensor [dim]
 * @param score Scratch buffer [head_num × seq_len]
 * @param pos Current position (attend to [0:pos])
 * @param layer_idx Current layer index
 * @param seq_len Maximum sequence length
 * @param kv_dim Key/Value dimension
 * @param head_num Number of query heads
 * @param head_size Dimension per head
 * @param kv_mul Query heads per KV head (for GQA)
 */
template <i32 BLOCK_SIZE>
__global__ void mha_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    float* __restrict__ score,
    i32 pos,
    i32 layer_idx,
    i32 seq_len,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 kv_mul);

// ============================================================================
// Host-side Launch Function
// ============================================================================

/**
 * @brief Launch multi-head attention CUDA kernel
 *
 * @param query Query tensor (device) [dim]
 * @param key_cache Key cache (device) [layer_num × seq_len × kv_dim]
 * @param value_cache Value cache (device) [layer_num × seq_len × kv_dim]
 * @param output Output tensor (device) [dim]
 * @param score Scratch buffer (device) [head_num × seq_len]
 * @param pos Current position
 * @param layer_idx Layer index
 * @param seq_len Sequence length
 * @param kv_dim KV dimension
 * @param head_num Number of heads
 * @param head_size Head dimension
 * @param kv_mul KV multiplier
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> mha_cuda_launch(
    std::span<const f32> query,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    std::span<f32> output,
    std::span<f32> score,
    i32 pos,
    i32 layer_idx,
    i32 seq_len,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 kv_mul,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_MHA_KERNEL_CUH
