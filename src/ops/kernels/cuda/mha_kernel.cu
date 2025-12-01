/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file mha_kernel.cu
 * @brief CUDA Multi-Head Attention kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/mha_kernel.cuh"
#include <cub/cub.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Device function to compute softmax in-place
 * Standard implementation
 */
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  const int thread_id = threadIdx.x;
  const int block_size = blockDim.x;

  // Find maximum value for numerical stability
  float local_max = (thread_id < size) ? x[thread_id] : 0.0f;
  for (int idx = thread_id + block_size; idx < size; idx += block_size) {
    if (x[idx] > local_max) {
      local_max = x[idx];
    }
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ BlockReduce::TempStorage temp_storage;
  __shared__ float block_max;
  local_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
  if (threadIdx.x == 0) {
    block_max = local_max;
  }
  __syncthreads();
  local_max = block_max;

  // Compute exponential and accumulate sum
  float local_sum = 0.0f;
  for (int idx = thread_id; idx < size; idx += block_size) {
    x[idx] = expf(x[idx] - local_max);
    local_sum += x[idx];
  }
  local_sum = BlockReduce(temp_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    block_max = local_sum;
  }
  __syncthreads();
  local_sum = block_max;

  // Apply normalization
  for (int idx = thread_id; idx < size; idx += block_size) {
    x[idx] /= local_sum;
  }
}

/**
 * @brief CUDA kernel for Multi-Head Attention
 * Standard implementation
 */
__global__ void multi_head_attention_kernel(
    i32 pos,
    i32 seq_len,
    float* query,
    float* score_ptr,
    float* output,
    float* key_cache,
    float* value_cache,
    i32 kv_dim,
    i32 kv_mul,
    i32 head_num,
    i32 head_size,
    i32 layer_offset) {

  const int head_idx = blockIdx.x;
  if (head_idx >= head_num) {
    return;
  }

  float* q_head = query + head_idx * head_size;
  float* attn_scores = score_ptr + head_idx * seq_len;
  const float inv_sqrt_head_size = 1.0f / sqrtf(static_cast<float>(head_size));
  const i32 kv_head_offset = (head_idx / kv_mul) * head_size;

  // Compute attention scores: Q·K^T
  const int thread_id = threadIdx.x;
  const int block_size = blockDim.x;
  for (int time_step = thread_id; time_step <= pos; time_step += block_size) {
    const i32 key_offset = layer_offset + time_step * kv_dim + kv_head_offset;
    float* k_head = key_cache + key_offset;

    float dot_product = 0.0f;
#pragma unroll
    for (int dim_idx = 0; dim_idx < head_size; dim_idx += 4) {
      const float4 k_vec = *reinterpret_cast<float4*>(k_head + dim_idx);
      const float4 q_vec = *reinterpret_cast<float4*>(q_head + dim_idx);
      // Accumulate dot product components
      dot_product = dot_product + (dim_idx < head_size ? k_vec.x * q_vec.x : 0.0f);
      dot_product = dot_product + (dim_idx + 1 < head_size ? k_vec.y * q_vec.y : 0.0f);
      dot_product = dot_product + (dim_idx + 2 < head_size ? k_vec.z * q_vec.z : 0.0f);
      dot_product = dot_product + (dim_idx + 3 < head_size ? k_vec.w * q_vec.w : 0.0f);
    }

    // Scale by inverse square root of head size
    attn_scores[time_step] = dot_product * inv_sqrt_head_size;
  }
  __syncthreads();

  // Apply softmax normalization
  softmax_gpu(attn_scores, pos + 1);
  __syncthreads();

  // Compute weighted sum of values
  float* out_head = output + head_idx * head_size;
  const int thread_id_v = threadIdx.x;
  const int block_size_v = blockDim.x;
  for (int dim_idx = thread_id_v; dim_idx < head_size; dim_idx += block_size_v) {
    float weighted_sum = 0.0f;
#pragma unroll
    for (int time_step = 0; time_step <= pos; time_step++) {
      const i32 value_offset = layer_offset + time_step * kv_dim + kv_head_offset;
      float* v_head = value_cache + value_offset;
      const float attn_weight = attn_scores[time_step];
      weighted_sum = weighted_sum + attn_weight * v_head[dim_idx];
    }
    out_head[dim_idx] = weighted_sum;
  }
}

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
    cudaStream_t stream) {

  // Calculate layer offset (using standard approach)
  i32 layer_offset = layer_index * seq_len * kv_dim;
  i32 thread_num = 128;

  // Launch configuration
  if (stream) {
    multi_head_attention_kernel<<<head_num, thread_num, 0, stream>>>(
        pos, seq_len,
        const_cast<float*>(query.data()),
        score.data(),
        mha_out.data(),
        const_cast<float*>(key_cache.data()),
        const_cast<float*>(value_cache.data()),
        kv_dim, kv_mul, head_num, head_size, layer_offset);
  } else {
    multi_head_attention_kernel<<<head_num, thread_num>>>(
        pos, seq_len,
        const_cast<float*>(query.data()),
        score.data(),
        mha_out.data(),
        const_cast<float*>(key_cache.data()),
        const_cast<float*>(value_cache.data()),
        kv_dim, kv_mul, head_num, head_size, layer_offset);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA MHA kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA MHA kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
