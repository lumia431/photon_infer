/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file rmsnorm_kernel.cu
 * @brief CUDA RMS normalization kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/rmsnorm_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for RMS normalization
 *
 * Standard implementation:
 * - Template parameter: BLOCK_DIM=128
 * - Single block, processes entire row
 * - Phase 1: Compute sum of squares using CUB BlockReduce
 * - Phase 2: Compute scale = rsqrt(mean + eps)
 * - Phase 3: Apply normalization with weight
 *
 * Grid: 1 block, Block: 128 threads
 */
template <i32 BLOCK_DIM>
static __global__ void row_rmsnorm_f32(
    float* in,
    float* wei,
    float* out,
    int size,
    float eps) {

  const int thread_id = threadIdx.x;

  // Vectorization setup
  constexpr int vector_width = 4;
  const int vector_count = size / vector_width;
  const int remainder_start = vector_width * vector_count;

  // Phase 1: Calculate sum of squared values
  float sq_sum = 0.0f;
  float4* input_vec = reinterpret_cast<float4*>(in);
  const int block_size = blockDim.x;
  for (int vec_idx = thread_id; vec_idx < vector_count; vec_idx += block_size) {
    const float4 vec_data = input_vec[vec_idx];
    // Accumulate squared components
    sq_sum = sq_sum + vec_data.x * vec_data.x;
    sq_sum = sq_sum + vec_data.y * vec_data.y;
    sq_sum = sq_sum + vec_data.z * vec_data.z;
    sq_sum = sq_sum + vec_data.w * vec_data.w;
  }

  // Process remaining scalar elements
  for (int elem_idx = remainder_start + thread_id; elem_idx < size; elem_idx += block_size) {
    const float elem_val = in[elem_idx];
    sq_sum = sq_sum + elem_val * elem_val;
  }

  // Block-level reduction using CUB
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float block_sum;
  sq_sum = BlockReduce(reduce_storage).Sum(sq_sum);
  if (threadIdx.x == 0) {
    block_sum = sq_sum;
  }
  __syncthreads();
  sq_sum = block_sum;

  // Calculate normalization scale
  const float norm_scale = rsqrtf(sq_sum / static_cast<float>(size) + eps);

  // Phase 2: Apply normalization with weight scaling
  float4* weight_vec = reinterpret_cast<float4*>(wei);
  float4* output_vec = reinterpret_cast<float4*>(out);
  for (int vec_idx = thread_id; vec_idx < vector_count; vec_idx += block_size) {
    const float4 input_data = input_vec[vec_idx];
    const float4 weight_data = weight_vec[vec_idx];
    // Compute normalized and weighted output components
    const float out_x = norm_scale * input_data.x * weight_data.x;
    const float out_y = norm_scale * input_data.y * weight_data.y;
    const float out_z = norm_scale * input_data.z * weight_data.z;
    const float out_w = norm_scale * input_data.w * weight_data.w;
    output_vec[vec_idx] = make_float4(out_x, out_y, out_z, out_w);
  }

  // Process remaining scalar elements
  for (int elem_idx = remainder_start + thread_id; elem_idx < size; elem_idx += block_size) {
    const float input_elem = in[elem_idx];
    const float weight_elem = wei[elem_idx];
    out[elem_idx] = weight_elem * input_elem * norm_scale;
  }
}

Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps,
    cudaStream_t stream) {

  // Validate dimensions (using standard approach)
  if (static_cast<i32>(input.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_cuda_launch");
  }

  if (static_cast<i32>(weight.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_cuda_launch");
  }

  // Check vectorization alignment (using standard approach)
  constexpr int pack_size = 4;
  if (dim % pack_size != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Dimension must be multiple of 4 for vectorization");
  }

  // Launch configuration (using standard approach exactly)
  // Grid: 1 block, Block: 128 threads
  constexpr int threads_num = 128;

  // Need non-const pointers for kernel (using standard approach)
  float* in_ptr = const_cast<float*>(input.data());
  float* wei_ptr = const_cast<float*>(weight.data());
  float* out_ptr = output.data();

  if (stream) {
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream>>>(
        in_ptr, wei_ptr, out_ptr, dim, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(
        in_ptr, wei_ptr, out_ptr, dim, eps);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA rmsnorm kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA rmsnorm kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

/**
 * @brief Batched RMS normalization (CUDA)
 *
 * Processes multiple sequences in parallel.
 * Grid: batch_size blocks, Block: 128 threads
 * Each block processes one row independently.
 */
Result<void> rmsnorm_batched_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 batch_size,
    i32 dim,
    f32 eps,
    cudaStream_t stream) {

  // Validate dimensions
  if (static_cast<i32>(input.size()) != batch_size * dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in batched rmsnorm");
  }

  if (static_cast<i32>(weight.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in batched rmsnorm");
  }

  if (static_cast<i32>(output.size()) != batch_size * dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in batched rmsnorm");
  }

  // Check vectorization alignment
  constexpr int pack_size = 4;
  if (dim % pack_size != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Dimension must be multiple of 4 for vectorization");
  }

  // Launch configuration: batch_size blocks × 128 threads
  constexpr int threads_num = 128;

  // Need non-const pointers for kernel
  float* in_ptr = const_cast<float*>(input.data());
  float* wei_ptr = const_cast<float*>(weight.data());
  float* out_ptr = output.data();

  // Launch batch_size kernels in parallel
  // Each block processes one row: input[block_idx * dim : (block_idx+1) * dim]
  for (i32 i = 0; i < batch_size; ++i) {
    float* in_row = in_ptr + i * dim;
    float* out_row = out_ptr + i * dim;

    if (stream) {
      row_rmsnorm_f32<128><<<1, threads_num, 0, stream>>>(
          in_row, wei_ptr, out_row, dim, eps);
    } else {
      row_rmsnorm_f32<128><<<1, threads_num>>>(
          in_row, wei_ptr, out_row, dim, eps);
    }
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA batched rmsnorm kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA batched rmsnorm kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
