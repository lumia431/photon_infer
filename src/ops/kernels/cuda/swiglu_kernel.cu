/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file swiglu_kernel.cu
 * @brief CUDA SwiGLU activation kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/swiglu_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for SwiGLU activation
 *
 * Standard implementation:
 * - Uses shared memory for input caching
 * - Computes swish(in1) = in1 * sigmoid(in1)
 * - Multiplies with in2: out = swish(in1) * in2
 */
__global__ void swiglu_kernel_cu_fp32(
    int size,
    const float* in1,
    const float* in2,
    float* out) {

  const int thread_id = threadIdx.x;
  const int block_offset = blockDim.x * blockIdx.x;
  const int global_idx = block_offset + thread_id;
  
  // Bounds check
  if (global_idx < size) {
    // Shared memory allocation
    extern __shared__ float shared_buffer[];
    float* shared_in1 = shared_buffer;
    float* shared_in2 = shared_buffer + blockDim.x;

    // Load input data into shared memory
    shared_in1[thread_id] = in1[global_idx];
    shared_in2[thread_id] = in2[global_idx];
    __syncthreads();

    // Compute Swish activation: x * sigmoid(x)
    const float input_val = shared_in1[thread_id];
    const float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
    const float swish_result = input_val * sigmoid_val;

    // Multiply Swish output with second input
    out[global_idx] = swish_result * shared_in2[thread_id];
  }
}

Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  // Validate dimensions (using standard approach)
  if (static_cast<i32>(input1.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input1 size mismatch in swiglu_cuda_launch");
  }

  if (static_cast<i32>(input2.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input2 size mismatch in swiglu_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in swiglu_cuda_launch");
  }

  // Launch configuration with alternative variable names
  constexpr int thread_count = 128;
  const int block_count = (size + thread_count - 1) / thread_count;
  const size_t shared_mem_size = thread_count * sizeof(float) * 2;

  if (stream) {
    swiglu_kernel_cu_fp32<<<block_count, thread_count, shared_mem_size, stream>>>(
        size, input1.data(), input2.data(), output.data());
  } else {
    swiglu_kernel_cu_fp32<<<block_count, thread_count, shared_mem_size>>>(
        size, input1.data(), input2.data(), output.data());
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA swiglu kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA swiglu kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

/**
 * @brief CUDA kernel for batched SwiGLU activation
 *
 * Processes all batch elements in parallel.
 * Grid: [batch_size × (hidden_dim + 127) / 128]
 */
__global__ void swiglu_batched_kernel_cu_fp32(
    i32 batch_size,
    i32 hidden_dim,
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out) {

  // Calculate batch index and element index within batch
  const i32 total_idx = threadIdx.x + blockDim.x * blockIdx.x;
  const i32 batch_idx = total_idx / hidden_dim;
  const i32 elem_idx = total_idx % hidden_dim;

  if (batch_idx >= batch_size || elem_idx >= hidden_dim) {
    return;
  }

  const i32 idx = batch_idx * hidden_dim + elem_idx;

  // Load inputs
  const f32 val1 = in1[idx];
  const f32 val2 = in2[idx];

  // Compute swish activation: x * sigmoid(x)
  const f32 sigmoid_val = 1.0f / (1.0f + expf(-val1));
  const f32 swish_val = val1 * sigmoid_val;

  // Multiply with second input
  out[idx] = swish_val * val2;
}

Result<void> swiglu_batched_cuda_launch(
    const f32* input1,
    const f32* input2,
    f32* output,
    i32 batch_size,
    i32 hidden_dim,
    cudaStream_t stream) {

  // Total elements to process
  const i32 total_elements = batch_size * hidden_dim;

  // Launch configuration
  constexpr i32 threads = 256;
  const i32 blocks = (total_elements + threads - 1) / threads;

  if (stream) {
    swiglu_batched_kernel_cu_fp32<<<blocks, threads, 0, stream>>>(
        batch_size, hidden_dim, input1, input2, output);
  } else {
    swiglu_batched_kernel_cu_fp32<<<blocks, threads>>>(
        batch_size, hidden_dim, input1, input2, output);
  }

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("Batched SwiGLU kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
