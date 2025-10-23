/**
 * @file swiglu_kernel.cu
 * @brief CUDA implementation of SwiGLU activation
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/swiglu_kernel.cuh"

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

template <i32 BLOCK_SIZE>
__global__ void swiglu_kernel(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    i32 size) {

  const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;
  const i32 tid = threadIdx.x;

  if (idx >= size) {
    return;
  }

  // Shared memory for caching inputs
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;               // First BLOCK_SIZE floats
  float* smem2 = shared_mem + BLOCK_SIZE;  // Next BLOCK_SIZE floats

  // Load inputs into shared memory
  smem1[tid] = input1[idx];
  smem2[tid] = input2[idx];
  __syncthreads();

  // Compute Swish(x) = x * sigmoid(x)
  // sigmoid(x) = 1 / (1 + exp(-x))
  const float x = smem1[tid];
  const float sigmoid_x = 1.0f / (1.0f + expf(-x));
  const float swish_x = x * sigmoid_x;

  // Compute SwiGLU: Swish(input1) * input2
  output[idx] = swish_x * smem2[tid];
}

// ============================================================================
// Optimized kernel without shared memory (for large sizes)
// ============================================================================

/**
 * @brief SwiGLU kernel without shared memory (direct computation)
 *
 * Faster for large arrays where shared memory overhead doesn't pay off.
 */
__global__ void swiglu_kernel_direct(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    i32 size) {

  const i32 idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= size) {
    return;
  }

  // Direct computation without shared memory
  const float x = input1[idx];
  const float sigmoid_x = 1.0f / (1.0f + expf(-x));
  const float swish_x = x * sigmoid_x;

  output[idx] = swish_x * input2[idx];
}

// ============================================================================
// Host-side Launch Function
// ============================================================================

Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  // Validate sizes
  if (input1.size() != static_cast<usize>(size) ||
      input2.size() != static_cast<usize>(size) ||
      output.size() != static_cast<usize>(size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Size mismatch in swiglu_cuda_launch");
  }

  constexpr i32 BLOCK_SIZE = 128;
  const i32 grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // For small sizes, use shared memory version
  // For large sizes, use direct version (avoids shared memory overhead)
  const bool use_shared_mem = (size < 100000);

  if (use_shared_mem) {
    const usize shmem_size = 2 * BLOCK_SIZE * sizeof(float);

    if (stream != nullptr) {
      swiglu_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, shmem_size, stream>>>(
          input1.data(), input2.data(), output.data(), size);
    } else {
      swiglu_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, shmem_size>>>(
          input1.data(), input2.data(), output.data(), size);
    }
  } else {
    // Direct version (no shared memory)
    if (stream != nullptr) {
      swiglu_kernel_direct<<<grid_size, BLOCK_SIZE, 0, stream>>>(
          input1.data(), input2.data(), output.data(), size);
    } else {
      swiglu_kernel_direct<<<grid_size, BLOCK_SIZE>>>(
          input1.data(), input2.data(), output.data(), size);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA swiglu kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

template __global__ void swiglu_kernel<128>(
    const float*, const float*, float*, i32);

}  // namespace photon::kernels::cuda
