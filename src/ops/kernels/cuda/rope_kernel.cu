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
__device__ void rope_calc(float fcr, float fci, float* vec, i32 idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr = make_float2(
      vec_value.x * fcr - vec_value.y * fci,
      vec_value.x * fci + vec_value.y * fcr);
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

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;  // Process 2 elements per thread
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  // Apply to Q
  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);

  // Apply to K if within kv_dim
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
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

  // Launch configuration (using standard approach)
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;

  if (stream) {
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream>>>(
        pos, dim, kv_dim, head_size,
        q.data(), k.data(),
        sin_cache.data(), cos_cache.data());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(
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
