/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file paged_kv_write.cu
 * @brief Optimized kernel for writing K/V to paged cache
 */

#include "photon/ops/kernels/cuda/paged_kv_write.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Kernel to write K/V to block-based paged cache
 *
 * Grid: (batch_size, num_kv_heads)
 * Block: min(head_size, 256) threads
 *
 * Each block processes one (sequence, kv_head) pair
 */
template <int BLOCK_THREADS>
__global__ void paged_kv_write_kernel(
    const float* __restrict__ k_src,      // [batch_size, kv_dim]
    const float* __restrict__ v_src,      // [batch_size, kv_dim]
    float* __restrict__ key_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
    float* __restrict__ value_cache,      // [num_blocks, num_kv_heads, block_size, head_size]
    const i32* __restrict__ block_table,  // [batch_size, max_num_blocks_per_seq]
    const i32* __restrict__ positions,    // [batch_size]
    i32 num_kv_heads,
    i32 head_size,
    i32 block_size,
    i32 max_num_blocks_per_seq) {

  const int seq_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int tid = threadIdx.x;

  // Get token position for this sequence
  const int token_pos = positions[seq_idx];

  // Determine which block and offset within block
  const int block_idx = token_pos / block_size;
  const int block_offset = token_pos % block_size;

  // Get physical block ID from block table
  const int physical_block = block_table[seq_idx * max_num_blocks_per_seq + block_idx];

  if (physical_block < 0) {
    // Invalid block (shouldn't happen in normal operation)
    return;
  }

  // Compute source offset in k_src/v_src
  // k_src shape: [batch_size, num_kv_heads * head_size]
  const int src_offset = seq_idx * num_kv_heads * head_size + kv_head_idx * head_size;
  const float* k_src_ptr = k_src + src_offset;
  const float* v_src_ptr = v_src + src_offset;

  // Compute destination offset in cache
  // Cache shape: [num_blocks, num_kv_heads, block_size, head_size]
  const int cache_offset = physical_block * num_kv_heads * block_size * head_size
                         + kv_head_idx * block_size * head_size
                         + block_offset * head_size;
  float* k_dst_ptr = key_cache + cache_offset;
  float* v_dst_ptr = value_cache + cache_offset;

  // Copy head_size elements (parallelized across threads)
  for (int i = tid; i < head_size; i += BLOCK_THREADS) {
    k_dst_ptr[i] = k_src_ptr[i];
    v_dst_ptr[i] = v_src_ptr[i];
  }
}

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
    cudaStream_t stream) {

  // Validate inputs
  if (batch_size <= 0 || num_kv_heads <= 0 || head_size <= 0 || block_size <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid dimensions for paged KV write");
  }

  // Configure kernel launch
  // Grid: one block per (sequence, kv_head) pair
  dim3 grid(batch_size, num_kv_heads);

  // Block: use enough threads to cover head_size
  const int block_threads = min(head_size, 256);

  // Launch kernel with template dispatch
  if (block_threads <= 64) {
    paged_kv_write_kernel<64><<<grid, 64, 0, stream>>>(
        k_src, v_src, key_cache, value_cache, block_table, positions,
        num_kv_heads, head_size, block_size, max_num_blocks_per_seq);
  } else if (block_threads <= 128) {
    paged_kv_write_kernel<128><<<grid, 128, 0, stream>>>(
        k_src, v_src, key_cache, value_cache, block_table, positions,
        num_kv_heads, head_size, block_size, max_num_blocks_per_seq);
  } else {
    paged_kv_write_kernel<256><<<grid, 256, 0, stream>>>(
        k_src, v_src, key_cache, value_cache, block_table, positions,
        num_kv_heads, head_size, block_size, max_num_blocks_per_seq);
  }

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    "Paged KV write kernel failed: " +
                    std::string(cudaGetErrorString(err)));
  }

  return Ok();
}

} // namespace photon::kernels::cuda
