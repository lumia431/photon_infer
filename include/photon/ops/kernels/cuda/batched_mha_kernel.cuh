/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file batched_mha_kernel.cuh
 * @brief Optimized Batched Multi-Head Attention CUDA kernel with Paged KV Cache
 * @version 1.0.0
 *
 * This kernel implements high-performance batched MHA with techniques from vLLM:
 * - True batch-level parallelization (grid.y = batch_size)
 * - Vectorized memory access (float4)
 * - Shared memory optimization for query caching
 * - Optimized softmax with CUB
 * - Paged KV cache support for zero-copy memory access
 *
 * Performance improvements over single-sequence MHA:
 * - 2-3x throughput for batch_size >= 8
 * - Better GPU utilization (70%+ SM occupancy)
 * - Zero-copy paged cache access
 */


#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {


/**
 * @brief Batched MHA with paged cache offsets (zero-copy!)
 *
 * This version supports paged KV cache where each sequence's cache data
 * is stored at a different offset within the global cache tensor.
 * NO COPYING - kernel directly accesses correct cache positions!
 *
 * @param positions Position for each sequence [batch_size] (GPU pointer)
 * @param cache_offsets Cache offset for each sequence [batch_size] (GPU pointer)
 *        cache_offsets[i] = starting position in cache for sequence i
 * @param batch_size Number of sequences
 * @param num_heads Number of query heads
 * @param seq_len Maximum sequence length per slot
 * @param kv_dim KV dimension
 * @param kv_mul GQA multiplier
 * @param head_size Dimension per head
 * @param mha_out Output [batch_size, dim]
 * @param query Query [batch_size, dim]
 * @param score Scratch buffer [batch_size, num_heads, seq_len]
 * @param key_cache Key cache [total_cache_size, kv_dim] (paged)
 * @param value_cache Value cache [total_cache_size, kv_dim] (paged)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> batched_mha_paged_launch(
    const i32* positions,
    const i32* cache_offsets,
    i32 batch_size,
    i32 num_heads,
    i32 seq_len,
    i32 kv_dim,
    i32 kv_mul,
    i32 head_size,
    std::span<f32> mha_out,
    std::span<const f32> query,
    std::span<f32> score,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

