/**
 * @file swiglu_kernel.cuh
 * @brief CUDA kernels for SwiGLU activation
 * @version 0.1.0
 *
 * SwiGLU(x, W1, W3) = Swish(x @ W1) ⊙ (x @ W3)
 * Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Declaration
// ============================================================================

/**
 * @brief CUDA kernel for SwiGLU activation
 *
 * Computes: output[i] = Swish(input1[i]) * input2[i]
 * where Swish(x) = x * sigmoid(x)
 *
 * Uses shared memory for optimization.
 *
 * Grid:  (size + BLOCK_SIZE - 1) / BLOCK_SIZE blocks
 * Block: BLOCK_SIZE threads
 * Shared memory: 2 * BLOCK_SIZE * sizeof(float)
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param input1 First input (for Swish) [size]
 * @param input2 Second input (gate) [size]
 * @param output Output [size]
 * @param size Number of elements
 */
template <i32 BLOCK_SIZE>
__global__ void swiglu_kernel(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    i32 size);

// ============================================================================
// Host-side Launch Function
// ============================================================================

/**
 * @brief Launch SwiGLU CUDA kernel
 *
 * @param input1 First input data (device) [size]
 * @param input2 Second input data (device) [size]
 * @param output Output data (device) [size]
 * @param size Number of elements
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH
