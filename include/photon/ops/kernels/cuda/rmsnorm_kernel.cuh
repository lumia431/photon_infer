/**
 * @file rmsnorm_kernel.cuh
 * @brief CUDA kernels for RMS normalization
 * @version 0.1.0
 *
 * Modern C++20/CUDA implementation with:
 * - Float4 vectorized memory access
 * - CUB block-level primitives
 * - Warp-level optimization
 * - Compile-time configuration
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Configuration Constants
// ============================================================================

/// Block size for RMSNorm (tuned for occupancy)
inline constexpr i32 RMSNORM_BLOCK_SIZE = 128;

/// Vectorization pack size (float4 = 4 floats)
inline constexpr i32 RMSNORM_PACK_SIZE = 4;

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * @brief CUDA kernel for single-vector RMS normalization
 *
 * Algorithm:
 * 1. Each thread computes partial sum of squares (vectorized with float4)
 * 2. Block-level reduction to get total sum
 * 3. Compute rsqrt = 1 / sqrt(mean + eps)
 * 4. Normalize and scale output (vectorized with float4)
 *
 * Grid:  1 block
 * Block: BLOCK_DIM threads
 *
 * @tparam BLOCK_DIM Number of threads per block
 * @param input Input vector [dim]
 * @param weight Scaling factors [dim]
 * @param output Output vector [dim]
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <i32 BLOCK_DIM>
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 dim,
    float eps);

/**
 * @brief CUDA kernel for batch RMS normalization
 *
 * Each block processes one vector in the batch.
 *
 * Grid:  batch_size blocks
 * Block: BLOCK_DIM threads
 *
 * @tparam BLOCK_DIM Number of threads per block
 * @param input Input matrix [batch_size × dim] (row-major)
 * @param weight Scaling factors [dim]
 * @param output Output matrix [batch_size × dim] (row-major)
 * @param batch_size Number of vectors in batch
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <i32 BLOCK_DIM>
__global__ void rmsnorm_batch_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 batch_size,
    i32 dim,
    float eps);

// ============================================================================
// Host-side Launch Functions
// ============================================================================

/**
 * @brief Launch RMSNorm CUDA kernel (single vector)
 *
 * @param input Input data pointer (device)
 * @param weight Weight data pointer (device)
 * @param output Output data pointer (device)
 * @param dim Feature dimension
 * @param eps Epsilon value
 * @param stream CUDA stream (nullptr = default stream)
 * @return Result indicating success or error
 */
Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch RMSNorm CUDA kernel (batch)
 *
 * @param input Input data pointer (device)
 * @param weight Weight data pointer (device)
 * @param output Output data pointer (device)
 * @param batch_size Number of vectors
 * @param dim Feature dimension
 * @param eps Epsilon value
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> rmsnorm_batch_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 batch_size,
    i32 dim,
    f32 eps,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH
