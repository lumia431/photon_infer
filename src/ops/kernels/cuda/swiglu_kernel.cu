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
