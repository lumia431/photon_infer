/**
 * @file mha_kernel.cu
 * @brief CUDA Multi-Head Attention kernel implementation
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/mha_kernel.cu
 */

#include "photon/ops/kernels/cuda/mha_kernel.cuh"
#include <cub/cub.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Device function to compute softmax in-place
 * Following KuiperInfer line-by-line
 */
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // Find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : 0;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  // Compute exp and sum
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  // Normalize
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

/**
 * @brief CUDA kernel for Multi-Head Attention
 * Following KuiperInfer line-by-line
 */
__global__ void multi_head_attention_kernel(
    int32_t pos,
    int32_t seq_len,
    float* query,
    float* score_ptr,
    float* output,
    float* key_cache,
    float* value_cache,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_num,
    int32_t head_size,
    int32_t layer_offset) {

  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  float* query_head = query + head * head_size;
  float* score_head = score_ptr + head * seq_len;
  float scale = 1.f / sqrtf(head_size);
  int32_t head_offset = (head / kv_mul) * head_size;

  // Compute attention scores: Q·K^T
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
#pragma unroll
    for (int i = 0; i < head_size; i += 4) {
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      if (i < head_size) {
        score += key_head_float4.x * query_head_float4.x;
      }
      if (i + 1 < head_size) {
        score += key_head_float4.y * query_head_float4.y;
      }
      if (i + 2 < head_size) {
        score += key_head_float4.z * query_head_float4.z;
      }
      if (i + 3 < head_size) {
        score += key_head_float4.w * query_head_float4.w;
      }
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  // Apply softmax
  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // Weighted sum of values
  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
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

  // Calculate layer offset (following KuiperInfer)
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  int32_t thread_num = 128;

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
