/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#pragma once

/**
 * @file paged_kv_write.cuh
 * @brief Optimized kernel for writing K/V to paged cache
 * @version 1.0.0
 */

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief Write batched K/V to block-based paged cache
 *
 * This kernel efficiently writes K and V tensors from a batched forward pass
 * into the block-based KV cache using the block table for addressing.
 *
 * @param k_src Source K tensor [batch_size, kv_dim]
 * @param v_src Source V tensor [batch_size, kv_dim]
 * @param key_cache Destination key cache [num_blocks, num_kv_heads, block_size, head_size]
 * @param value_cache Destination value cache [num_blocks, num_kv_heads, block_size, head_size]
 * @param block_table Block table [batch_size, max_num_blocks_per_seq]
 * @param positions Token positions for each sequence [batch_size]
 * @param batch_size Number of sequences
 * @param num_kv_heads Number of KV heads
 * @param head_size Size of each head
 * @param block_size Tokens per block
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> paged_kv_write_launch(
    const float* k_src,
    const float* v_src,
    float* key_cache,
    float* value_cache,
    const i32* block_table,
    const i32* positions,
    i32 batch_size,
    i32 num_kv_heads,
    i32 head_size,
    i32 block_size,
    i32 max_num_blocks_per_seq,
    cudaStream_t stream = 0);

} // namespace photon::kernels::cuda
