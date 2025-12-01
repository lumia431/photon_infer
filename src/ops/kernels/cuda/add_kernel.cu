/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file add_kernel.cu
 * @brief CUDA element-wise addition kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/add_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for element-wise addition
 *
 * Standard implementation:
 * - Each thread processes one element
 * - Simple element-wise: out[i] = in1[i] + in2[i]
 */
__global__ void add_kernel_cu_fp32(
    i32 size,
    const float* in1,
    const float* in2,
    float* out) {

  // Calculate thread index: block offset + thread within block
  const i32 block_offset = blockIdx.x * blockDim.x;
  const i32 global_idx = block_offset + threadIdx.x;
  
  // Bounds check: only process valid indices
  if (global_idx < size) {
    // Perform element-wise addition
    out[global_idx] = in1[global_idx] + in2[global_idx];
  }
}

Result<void> add_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  // Validate dimensions (using standard approach)
  if (static_cast<i32>(input1.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input1 size mismatch in add_cuda_launch");
  }

  if (static_cast<i32>(input2.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input2 size mismatch in add_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != size) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in add_cuda_launch");
  }

  // Launch configuration with alternative calculation
  // Block: 512 threads, Grid: ceil(size / 512) blocks
  constexpr i32 thread_num = 512;
  i32 block_num = (size + thread_num - 1) / thread_num;

  if (stream) {
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream>>>(
        size, input1.data(), input2.data(), output.data());
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(
        size, input1.data(), input2.data(), output.data());
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA add kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA add kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
