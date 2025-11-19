/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file paged_attention.cu
 * @brief PagedAttention kernel implementation
 *
 * Key optimizations:
 * 1. Block-table based access for non-contiguous KV cache
 * 2. Shared memory for query caching
 * 3. CUB for efficient softmax reduction
 * 4. Vectorized loads where possible
 */

#include "photon/ops/kernels/cuda/paged_attention.cuh"
#include <cub/cub.cuh>
#include <glog/logging.h>
#include <cfloat>

namespace photon::kernels::cuda {

/**
 * @brief Compute softmax in-place using CUB
 */
template <int BLOCK_SIZE>
__device__ void softmax_inplace(float* __restrict__ logits, int size) {
  const int tid = threadIdx.x;

  // Step 1: Find max for numerical stability
  float max_val = -FLT_MAX;
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    max_val = fmaxf(max_val, logits[i]);
  }

  // Reduce to get global max
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max());

  __shared__ float shared_max;
  if (tid == 0) {
    shared_max = max_val;
  }
  __syncthreads();
  max_val = shared_max;

  // Step 2: Compute exp and sum
  float sum = 0.0f;
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    float val = expf(logits[i] - max_val);
    logits[i] = val;
    sum += val;
  }

  // Reduce to get global sum
  __syncthreads();  // Reuse temp_storage
  sum = BlockReduce(temp_storage).Sum(sum);

  __shared__ float shared_sum;
  if (tid == 0) {
    shared_sum = sum;
  }
  __syncthreads();
  sum = shared_sum;

  // Step 3: Normalize
  float inv_sum = 1.0f / (sum + 1e-6f);
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    logits[i] *= inv_sum;
  }
  __syncthreads();
}

/**
 * @brief PagedAttention V1 kernel
 *
 * Each block processes one (head, sequence) pair.
 * Grid: (num_heads, num_seqs)
 * Block: BLOCK_SIZE threads (128)
 *
 * Template parameters:
 * - BLOCK_SIZE: Number of threads per block
 * - HEAD_SIZE: Size of attention head (must be known at compile time)
 * - BLOCK_SIZE_TOKENS: Number of tokens per KV cache block
 */
template <int BLOCK_SIZE, int HEAD_SIZE, int BLOCK_SIZE_TOKENS>
__global__ void paged_attention_v1_kernel(
    float* __restrict__ output,           // [num_seqs, num_heads, head_size]
    const float* __restrict__ query,      // [num_seqs, num_heads, head_size]
    const float* __restrict__ key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    const float* __restrict__ value_cache, // [num_blocks, num_kv_heads, block_size, head_size]
    const i32* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const i32* __restrict__ seq_lens,     // [num_seqs]
    i32 num_kv_heads,
    i32 max_num_blocks_per_seq,
    float scale) {

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;

  // Get sequence length
  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  // Compute KV head index (for GQA)
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  // Get block table for this sequence
  const i32* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // Shared memory for query and attention scores
  __shared__ float shared_query[HEAD_SIZE];
  extern __shared__ float shared_logits[];  // Dynamic size based on seq_len

  // Load query to shared memory
  const float* query_ptr = query + (seq_idx * num_heads + head_idx) * HEAD_SIZE;
  for (int i = tid; i < HEAD_SIZE; i += BLOCK_SIZE) {
    shared_query[i] = query_ptr[i];
  }
  __syncthreads();

  // Compute Q·K^T for all tokens using block table
  const int num_blocks = (seq_len + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;

  for (int token_idx = tid; token_idx < seq_len; token_idx += BLOCK_SIZE) {
    // Determine which block and offset within block
    const int block_idx = token_idx / BLOCK_SIZE_TOKENS;
    const int block_offset = token_idx % BLOCK_SIZE_TOKENS;

    // Get physical block ID from block table
    const int physical_block = block_table[block_idx];
    if (physical_block < 0) {
      // Invalid block (padding)
      shared_logits[token_idx] = -FLT_MAX;
      continue;
    }

    // Compute address of key vector
    // key_cache shape: [num_blocks, num_kv_heads, block_size, head_size]
    const float* key_ptr = key_cache
        + physical_block * num_kv_heads * BLOCK_SIZE_TOKENS * HEAD_SIZE
        + kv_head_idx * BLOCK_SIZE_TOKENS * HEAD_SIZE
        + block_offset * HEAD_SIZE;

    // Compute dot product: Q · K
    float qk_dot = 0.0f;
    for (int i = 0; i < HEAD_SIZE; ++i) {
      qk_dot += shared_query[i] * key_ptr[i];
    }

    // Apply scale and store
    shared_logits[token_idx] = qk_dot * scale;
  }
  __syncthreads();

  // Compute softmax
  softmax_inplace<BLOCK_SIZE>(shared_logits, seq_len);

  // Compute weighted sum: O = sum(attention_weights * V)
  float* output_ptr = output + (seq_idx * num_heads + head_idx) * HEAD_SIZE;

  for (int i = tid; i < HEAD_SIZE; i += BLOCK_SIZE) {
    float acc = 0.0f;

    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
      // Determine block and offset
      const int block_idx = token_idx / BLOCK_SIZE_TOKENS;
      const int block_offset = token_idx % BLOCK_SIZE_TOKENS;

      // Get physical block
      const int physical_block = block_table[block_idx];
      if (physical_block < 0) continue;

      // Compute address of value vector
      // value_cache shape: [num_blocks, num_kv_heads, block_size, head_size]
      const float* value_ptr = value_cache
          + physical_block * num_kv_heads * BLOCK_SIZE_TOKENS * HEAD_SIZE
          + kv_head_idx * BLOCK_SIZE_TOKENS * HEAD_SIZE
          + block_offset * HEAD_SIZE;

      // Accumulate weighted value
      acc += shared_logits[token_idx] * value_ptr[i];
    }

    output_ptr[i] = acc;
  }
}

/**
 * @brief Launch wrapper for paged attention V1
 */
Result<void> paged_attention_launch(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const i32* block_tables,
    const i32* seq_lens,
    i32 num_seqs,
    i32 num_heads,
    i32 num_kv_heads,
    i32 head_size,
    i32 block_size,
    i32 max_num_blocks_per_seq,
    float scale,
    cudaStream_t stream) {

  // Validate parameters
  if (num_seqs <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_size <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid dimensions for paged attention");
  }

  if (num_heads % num_kv_heads != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "num_heads must be divisible by num_kv_heads (GQA)");
  }

  // Grid and block configuration
  constexpr int BLOCK_SIZE = 128;
  dim3 grid(num_heads, num_seqs);
  dim3 block(BLOCK_SIZE);

  // Calculate shared memory size
  // Need space for attention scores (max sequence length = 4096 for safety)
  const int max_seq_len = 4096;
  size_t shared_mem_size = max_seq_len * sizeof(float);

  // Template dispatch based on head_size and block_size
  // Common head sizes: 64, 80, 96, 128
  // Common block sizes: 16
#define LAUNCH_KERNEL(HEAD_SIZE, BLOCK_SIZE_TOKENS)                           \
  paged_attention_v1_kernel<BLOCK_SIZE, HEAD_SIZE, BLOCK_SIZE_TOKENS>        \
      <<<grid, block, shared_mem_size, stream>>>(                             \
          output, query, key_cache, value_cache, block_tables, seq_lens,      \
          num_kv_heads, max_num_blocks_per_seq, scale);

  if (block_size == 16) {
    switch (head_size) {
      case 64:
        LAUNCH_KERNEL(64, 16);
        break;
      case 80:
        LAUNCH_KERNEL(80, 16);
        break;
      case 96:
        LAUNCH_KERNEL(96, 16);
        break;
      case 128:
        LAUNCH_KERNEL(128, 16);
        break;
      default:
        return Err<void>(ErrorCode::InvalidArgument,
                        "Unsupported head_size: " + std::to_string(head_size) +
                        ". Supported: 64, 80, 96, 128");
    }
  } else {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Unsupported block_size: " + std::to_string(block_size) +
                    ". Only 16 is supported currently");
  }

#undef LAUNCH_KERNEL

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    "PagedAttention V1 kernel launch failed: " +
                    std::string(cudaGetErrorString(err)));
  }

  return Ok();
}

} // namespace photon::kernels::cuda
