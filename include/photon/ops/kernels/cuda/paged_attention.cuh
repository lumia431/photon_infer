/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#pragma once

/**
 * @file paged_attention.cuh
 * @brief PagedAttention kernel for block-based KV cache
 * @version 1.0.0
 *
 * Implements paged attention with block table for non-contiguous memory access.
 * Inspired by vLLM's PagedAttention implementation.
 */

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief Launch PagedAttention V1 kernel
 *
 * Computes attention output using block-based paged KV cache.
 *
 * Grid: dim3(num_heads, num_seqs)
 * Block: 128 threads
 *
 * @param output [num_seqs, num_heads, head_size]
 * @param query [num_seqs, num_heads, head_size]
 * @param key_cache [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache [num_blocks, num_kv_heads, block_size, head_size]
 * @param block_tables [num_seqs, max_num_blocks_per_seq] - CPU tensor
 * @param seq_lens [num_seqs] - number of tokens per sequence
 * @param num_seqs Number of sequences in batch
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param head_size Size of each attention head
 * @param block_size Number of tokens per block
 * @param max_num_blocks_per_seq Maximum blocks per sequence
 * @param scale Attention scale factor (typically 1/sqrt(head_size))
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> paged_attention_launch(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const i32* block_tables,      // On GPU
    const i32* seq_lens,          // On GPU
    i32 num_seqs,
    i32 num_heads,
    i32 num_kv_heads,
    i32 head_size,
    i32 block_size,
    i32 max_num_blocks_per_seq,
    float scale,
    cudaStream_t stream = 0);

} // namespace photon::kernels::cuda
