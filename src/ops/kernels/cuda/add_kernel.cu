/**
 * @file add_kernel.cu
 * @brief CUDA element-wise addition kernel implementation
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/add_kernel.cu
 */

#include "photon/ops/kernels/cuda/add_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for element-wise addition
 *
 * Following KuiperInfer line-by-line:
 * - Each thread processes one element
 * - Simple element-wise: out[i] = in1[i] + in2[i]
 */
__global__ void add_kernel_cu_fp32(
    int32_t size,
    const float* in1,
    const float* in2,
    float* out) {

  // Following KuiperInfer: compute global thread ID
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }

  // Element-wise addition (following KuiperInfer)
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

Result<void> add_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  // Validate dimensions (following KuiperInfer)
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

  // Launch configuration (following KuiperInfer exactly)
  // Block: 512 threads, Grid: (size + 512 - 1) / 512 blocks
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
