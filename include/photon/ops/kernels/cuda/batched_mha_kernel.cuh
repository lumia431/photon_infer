/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file batched_mha_kernel.cuh
 * @brief Optimized Batched Multi-Head Attention CUDA kernel
 * @version 2.0.0
 *
 * This kernel implements high-performance batched MHA with techniques from vLLM:
 * - True batch-level parallelization (grid.y = batch_size)
 * - Vectorized memory access (float4)
 * - Shared memory optimization for query caching
 * - Optimized softmax with CUB
 * - **NEW v2.0:** Partitioned attention for long sequences (1.3-1.8x speedup)
 *
 * Performance improvements over single-sequence MHA:
 * - 2-3x throughput for batch_size >= 8
 * - Better GPU utilization (70%+ SM occupancy)
 * - 1.3-1.8x speedup for sequences > 256 tokens (with partitioning)
 */


#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch optimized batched Multi-Head Attention kernel
 *
 * Processes multiple sequences in parallel with optimized memory access patterns.
 *
 * **Input tensor layouts:**
 * - query:       [batch_size, num_heads, head_size]
 * - key_cache:   [num_layers, seq_len, kv_dim] (contiguous layout)
 * - value_cache: [num_layers, seq_len, kv_dim]
 *
 * **Grid/Block configuration:**
 * - Grid:  dim3(num_heads, batch_size, 1)  ← Key difference from single-seq version!
 * - Block: 128 threads
 *
 * **Algorithm:**
 * For each (seq_idx, head_idx) pair in parallel:
 * 1. Load query[seq_idx, head_idx] to shared memory
 * 2. Compute QK^T with vectorized loads (float4)
 * 3. Apply softmax using CUB BlockReduce
 * 4. Compute weighted sum of values
 *
 * @param positions Position for each sequence in batch [batch_size]
 * @param batch_size Number of sequences in batch
 * @param num_heads Number of query heads
 * @param layer_index Current layer index
 * @param seq_len Maximum sequence length
 * @param kv_dim Key/Value dimension (num_kv_heads * head_size)
 * @param kv_mul Query-to-KV head ratio (for GQA: num_heads / num_kv_heads)
 * @param head_size Dimension per head
 * @param mha_out Output [batch_size, num_heads, head_size]
 * @param query Query [batch_size, num_heads, head_size]
 * @param score Scratch buffer [batch_size, num_heads, seq_len]
 * @param key_cache Key cache [num_layers, seq_len, kv_dim]
 * @param value_cache Value cache [num_layers, seq_len, kv_dim]
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> batched_mha_cuda_launch(
    const i32* positions,
    i32 batch_size,
    i32 num_heads,
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

/**
 * @brief Launch partitioned attention kernels for long sequences (v2.0)
 *
 * Automatically partitions long sequences (> 256 tokens) into chunks of 512 tokens.
 * Each partition is processed in parallel by separate thread blocks, then merged
 * using a numerically stable reduce kernel.
 *
 * **Performance characteristics:**
 * - 1.3-1.8x speedup for sequences with 512-2048 tokens
 * - Best for decode phase with long contexts
 * - Automatic partition count selection based on max sequence length
 *
 * **Algorithm:**
 * 1. Partition kernel: Each (head, seq, partition) computes partial softmax
 * 2. Reduce kernel: Merges partitions with numerically stable method
 *
 * @param positions Position for each sequence [batch_size] (GPU pointer)
 * @param batch_size Number of sequences
 * @param num_heads Number of query heads
 * @param layer_index Current layer
 * @param seq_len Maximum sequence length
 * @param kv_dim KV dimension
 * @param kv_mul GQA multiplier
 * @param head_size Dimension per head
 * @param mha_out Output [batch_size, num_heads, head_size]
 * @param query Query [batch_size, num_heads, head_size]
 * @param key_cache Key cache [num_layers, seq_len, kv_dim]
 * @param value_cache Value cache [num_layers, seq_len, kv_dim]
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> partitioned_mha_cuda_launch(
    const i32* positions,
    i32 batch_size,
    i32 num_heads,
    i32 layer_index,
    i32 seq_len,
    i32 kv_dim,
    i32 kv_mul,
    i32 head_size,
    std::span<f32> mha_out,
    std::span<const f32> query,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    cudaStream_t stream = nullptr);

/**
 * @brief Single-sequence partitioned attention (convenience wrapper)
 *
 * Automatically uses partitioned kernel for long sequences.
 * Same interface as mha_cuda_launch but uses partitioning when beneficial.
 *
 * @param pos Current position
 * @param num_heads Number of query heads
 * @param layer_index Layer index
 * @param seq_len Maximum sequence length
 * @param kv_dim KV dimension
 * @param kv_mul GQA multiplier
 * @param head_size Dimension per head
 * @param mha_out Output [num_heads, head_size]
 * @param query Query [num_heads, head_size]
 * @param key_cache Key cache [seq_len, kv_dim]
 * @param value_cache Value cache [seq_len, kv_dim]
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> partitioned_mha_single_seq_launch(
    i32 pos,
    i32 num_heads,
    i32 layer_index,
    i32 seq_len,
    i32 kv_dim,
    i32 kv_mul,
    i32 head_size,
    std::span<f32> mha_out,
    std::span<const f32> query,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    cudaStream_t stream = nullptr);

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

