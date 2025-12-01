/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file mha_kernel.cuh
 * @brief CUDA Multi-Head Attention kernel interface
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */


#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA Multi-Head Attention kernel
 *
 * Following standard design:
 * - Computes attention: softmax(Q·K^T / √d) · V
 * - Supports Grouped Query Attention (GQA)
 * - Grid: head_num blocks (one per head)
 * - Block: 128 threads
 * - Uses float4 vectorization and CUB BlockReduce
 *
 * Algorithm per head:
 * 1. Compute scores: score[t] = query • key_cache[t] for t ∈ [0, pos]
 * 2. Apply softmax to scores
 * 3. Weighted sum: output = Σ(score[t] * value_cache[t])
 *
 * @param pos Current position in sequence
 * @param head_num Number of query heads
 * @param layer_index Current layer index (for cache offset)
 * @param seq_len Maximum sequence length
 * @param kv_dim Key/Value dimension
 * @param kv_mul Query-to-KV head ratio (for GQA)
 * @param head_size Dimension per head
 * @param mha_out Output tensor [head_num × head_size]
 * @param query Query tensor [head_num × head_size]
 * @param score Temporary score buffer [head_num × seq_len]
 * @param key_cache Key cache [num_layers × seq_len × kv_dim]
 * @param value_cache Value cache [num_layers × seq_len × kv_dim]
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> mha_cuda_launch(
    i32 pos,
    i32 head_num,
    i32 layer_index,
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

