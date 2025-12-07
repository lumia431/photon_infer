/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file batched_mha_kernel.cu
 * @brief Optimized Batched Multi-Head Attention CUDA implementation with Paged KV Cache
 * @version 1.0.0
 *
 * Key optimizations inspired by vLLM:
 * 1. Batch-level parallelization: grid.y = batch_size
 * 2. Vectorized loads: float4 for K/V access
 * 3. Shared memory: cache query vector
 * 4. Optimized softmax: CUB BlockReduce
 * 5. Paged KV cache: Zero-copy access with cache offsets
 */

#include "photon/ops/kernels/cuda/batched_mha_kernel.cuh"
#include <cub/cub.cuh>
#include <glog/logging.h>
#include <cfloat>  // for FLT_MAX

namespace photon::kernels::cuda {

// ============================================================================
// Helper: Softmax with CUB (numerically stable)
// ============================================================================

/**
 * @brief Compute softmax in-place using CUB for efficient reduction
 */
template <int BLOCK_SIZE>
__device__ void softmax_inplace(float* __restrict__ x, int size) {
  const int tid = threadIdx.x;

  // Step 1: Find max (for numerical stability)
  float max_val = -FLT_MAX;
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    max_val = fmaxf(max_val, x[i]);
  }

  // Block-level reduction to get global max
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
    float val = expf(x[i] - max_val);
    x[i] = val;
    sum += val;
  }

  // Block-level sum reduction
  __syncthreads();  // Reuse temp_storage
  sum = BlockReduce(temp_storage).Sum(sum);

  __shared__ float shared_sum;
  if (tid == 0) {
    shared_sum = sum;
  }
  __syncthreads();
  sum = shared_sum;

  // Step 3: Normalize
  float inv_sum = 1.0f / sum;
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    x[i] *= inv_sum;
  }
}

// ============================================================================
// Batched MHA Kernel with Paged Cache Support
// ============================================================================

/**
 * @brief Optimized batched multi-head attention kernel with paged cache offsets
 *
 * Grid:  dim3(num_heads, batch_size)  ← Each (head, seq) pair gets one block
 * Block: BLOCK_SIZE threads (128)
 *
 * This version supports paged KV cache where each sequence's data is stored
 * at a different offset in the global cache.
 *
 * @param cache_offsets [batch_size] - Starting position in cache for each sequence
 */
template <int BLOCK_SIZE, int HEAD_SIZE>
__global__ void batched_mha_paged_kernel(
    const i32* __restrict__ positions,      // [batch_size]
    const i32* __restrict__ cache_offsets,  // [batch_size] - NEW!
    const float* __restrict__ query,            // [batch_size, num_heads, head_size]
    float* __restrict__ score,                  // [batch_size, num_heads, seq_len]
    float* __restrict__ output,                 // [batch_size, num_heads, head_size]
    const float* __restrict__ key_cache,        // [total_cache_size, kv_dim] - paged!
    const float* __restrict__ value_cache,      // [total_cache_size, kv_dim] - paged!
    i32 seq_len,
    i32 kv_dim,
    i32 kv_mul,
    i32 num_heads,
    i32 head_size) {

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (head_idx >= num_heads) return;

  // Get position and cache offset for this sequence
  const int pos = positions[seq_idx];
  const int cache_offset = cache_offsets[seq_idx];  // NEW: sequence-specific offset
  const int num_tokens = pos + 1;

  // Compute pointers
  const int batch_head_offset = seq_idx * num_heads + head_idx;
  const float* query_ptr = query + batch_head_offset * head_size;
  float* score_ptr = score + batch_head_offset * seq_len;
  float* output_ptr = output + batch_head_offset * head_size;

  const int kv_head_idx = head_idx / kv_mul;
  const int kv_head_offset = kv_head_idx * head_size;
  const float scale = rsqrtf(static_cast<float>(head_size));

  // Cache query in shared memory
  __shared__ float shared_query[HEAD_SIZE];

  constexpr int VEC_SIZE = 4;
  constexpr int NUM_VECS = HEAD_SIZE / VEC_SIZE;

  if constexpr (HEAD_SIZE % VEC_SIZE == 0) {
    for (int i = tid; i < NUM_VECS; i += BLOCK_SIZE) {
      float4 q_vec = reinterpret_cast<const float4*>(query_ptr)[i];
      reinterpret_cast<float4*>(shared_query)[i] = q_vec;
    }
  } else {
    for (int i = tid; i < head_size; i += BLOCK_SIZE) {
      shared_query[i] = query_ptr[i];
    }
  }
  __syncthreads();

  // Step 1: Compute attention scores Q·K^T
  for (int t = tid; t < num_tokens; t += BLOCK_SIZE) {
    // NEW: Use cache_offset to access correct cache position
    const float* key_ptr = key_cache + (cache_offset + t) * kv_dim + kv_head_offset;

    float qk_dot = 0.0f;

    if constexpr (HEAD_SIZE % VEC_SIZE == 0) {
      #pragma unroll
      for (int i = 0; i < NUM_VECS; ++i) {
        float4 k_vec = reinterpret_cast<const float4*>(key_ptr)[i];
        float4 q_vec = reinterpret_cast<const float4*>(shared_query)[i];

        qk_dot += k_vec.x * q_vec.x;
        qk_dot += k_vec.y * q_vec.y;
        qk_dot += k_vec.z * q_vec.z;
        qk_dot += k_vec.w * q_vec.w;
      }
    } else {
      for (int i = 0; i < head_size; ++i) {
        qk_dot += key_ptr[i] * shared_query[i];
      }
    }

    score_ptr[t] = qk_dot * scale;
  }
  __syncthreads();

  // Step 2: Apply softmax
  softmax_inplace<BLOCK_SIZE>(score_ptr, num_tokens);
  __syncthreads();

  // Step 3: Weighted sum of values
  for (int i = tid; i < head_size; i += BLOCK_SIZE) {
    float acc = 0.0f;

    for (int t = 0; t < num_tokens; ++t) {
      // NEW: Use cache_offset to access correct cache position
      const float* value_ptr = value_cache + (cache_offset + t) * kv_dim + kv_head_offset;
      const float attn_weight = score_ptr[t];
      acc += attn_weight * value_ptr[i];
    }

    output_ptr[i] = acc;
  }
}


// ============================================================================
// Partitioned Attention (v2.0) - vLLM-inspired
// ============================================================================
/**
 * @brief Batched MHA with paged cache support (zero-copy!)
 *
 * This version accepts cache_offsets so each sequence can read from its
 * correct position in the paged cache without copying.
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
    cudaStream_t stream) {

  dim3 grid(num_heads, batch_size);
  dim3 block(128);

  auto launch_kernel = [&]<int HEAD_SIZE>() {
    if (stream) {
      batched_mha_paged_kernel<128, HEAD_SIZE>
          <<<grid, block, 0, stream>>>(
              positions,
              cache_offsets,
              query.data(),
              score.data(),
              mha_out.data(),
              key_cache.data(),
              value_cache.data(),
              seq_len, kv_dim, kv_mul, num_heads, head_size);
    } else {
      batched_mha_paged_kernel<128, HEAD_SIZE>
          <<<grid, block>>>(
              positions,
              cache_offsets,
              query.data(),
              score.data(),
              mha_out.data(),
              key_cache.data(),
              value_cache.data(),
              seq_len, kv_dim, kv_mul, num_heads, head_size);
    }
  };

  switch (head_size) {
    case 64:
      launch_kernel.template operator()<64>();
      break;
    case 80:
      launch_kernel.template operator()<80>();
      break;
    case 96:
      launch_kernel.template operator()<96>();
      break;
    case 128:
      launch_kernel.template operator()<128>();
      break;
    default:
      LOG(ERROR) << "Unsupported head_size: " << head_size;
      return Err<void>(ErrorCode::InvalidArgument,
                      "Unsupported head_size for batched paged MHA");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "Batched paged MHA kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("Batched paged MHA kernel failed: ") +
                    cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
