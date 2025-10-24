/**
 * @file swiglu_kernel.cu
 * @brief CUDA SwiGLU activation kernel implementation
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/swiglu_kernel.cu
 */

#include "photon/ops/kernels/cuda/swiglu_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for SwiGLU activation
 *
 * Following KuiperInfer line-by-line:
 * - Uses shared memory for input caching
 * - Computes swish(in1) = in1 * sigmoid(in1)
 * - Multiplies with in2: out = swish(in1) * in2
 */
__global__ void swiglu_kernel_cu_fp32(
    int size,
    const float* in1,
    const float* in2,
    float* out) {

  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }

  // Following KuiperInfer: shared memory layout
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  // Load inputs to shared memory
  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  // Compute swish activation: x * sigmoid(x)
  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  // Multiply with second input
  out[idx] = smem1[tid] * smem2[tid];
}

Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  // Validate dimensions (following KuiperInfer)
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

  // Launch configuration (following KuiperInfer exactly)
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = threads * sizeof(float) * 2;

  if (stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream>>>(
        size, input1.data(), input2.data(), output.data());
  } else {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
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

}  // namespace photon::kernels::cuda
