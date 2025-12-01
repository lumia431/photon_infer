/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file rope_kernel.cu
 * @brief CUDA RoPE (Rotary Position Embedding) kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/rope_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Device function to apply RoPE rotation
 * Standard implementation
 */
__device__ void rope_calc(float cos_val, float sin_val, float* vec, i32 idx) {
  float2* vec2_ptr = reinterpret_cast<float2*>(vec + idx);
  const float2 original = *vec2_ptr;
  // Apply rotation: [cos -sin] [x]   [x*cos - y*sin]
  //                 [sin  cos] [y] = [x*sin + y*cos]
  const float new_x = original.x * cos_val - original.y * sin_val;
  const float new_y = original.x * sin_val + original.y * cos_val;
  *vec2_ptr = make_float2(new_x, new_y);
}

/**
 * @brief CUDA kernel for RoPE application
 * Standard implementation
 */
__global__ void rope_kernel_cu_fp32(
    int pos,
    int dim,
    int kv_dim,
    int head_size,
    const float* input_q,
    const float* input_k,
    const float* sin_cache,
    const float* cos_cache) {

  const int base_idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int vec_idx = base_idx * 2;  // Process 2 elements per thread
  if (vec_idx >= dim) {
    return;
  }

  const int dim_in_head = vec_idx % head_size;
  const float sin_val = sin_cache[pos * head_size + dim_in_head];
  const float cos_val = cos_cache[pos * head_size + dim_in_head];

  // Apply rotation to query
  rope_calc(cos_val, sin_val, const_cast<float*>(input_q), vec_idx);

  // Apply rotation to key if within valid range
  if (vec_idx >= kv_dim) {
    return;
  }
  rope_calc(cos_val, sin_val, const_cast<float*>(input_k), vec_idx);
}


Result<void> rope_cuda_launch(
    std::span<f32> q,
    std::span<f32> k,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream) {

  // Validate inputs
  if (static_cast<i32>(q.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Query size mismatch in rope_cuda_launch");
  }

  if (static_cast<i32>(k.size()) != kv_dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Key size mismatch in rope_cuda_launch");
  }

  // Launch configuration with alternative calculation
  constexpr int thread_count = 128;
  const int block_count = (dim + thread_count - 1) / thread_count;

  if (stream) {
    rope_kernel_cu_fp32<<<block_count, thread_count, 0, stream>>>(
        pos, dim, kv_dim, head_size,
        q.data(), k.data(),
        sin_cache.data(), cos_cache.data());
  } else {
    rope_kernel_cu_fp32<<<block_count, thread_count>>>(
        pos, dim, kv_dim, head_size,
        q.data(), k.data(),
        sin_cache.data(), cos_cache.data());
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA rope kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA rope failed: ") + cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
