/**
 * @file matmul_kernel.cuh
 * @brief CUDA kernels for matrix multiplication
 * @version 0.1.0
 *
 * Modern C++20/CUDA implementation with:
 * - Float4 vectorized memory access
 * - CUB block-level reduction
 * - Shared memory optimization
 * - Support for GEMV (matrix-vector) and GEMM (matrix-matrix)
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Configuration Constants
// ============================================================================

/// Block size for MatMul
inline constexpr i32 MATMUL_BLOCK_SIZE = 128;

/// Rows processed per block
inline constexpr i32 MATMUL_ROWS_PER_BLOCK = 1;

/// Vectorization pack size
inline constexpr i32 MATMUL_PACK_SIZE = 4;

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * @brief CUDA kernel for matrix-vector multiplication (GEMV)
 *
 * Computes: output[M] = weight[M × N] @ input[N]
 *
 * Each block computes one output row.
 * Within a block, threads collaboratively compute the dot product.
 *
 * Grid:  (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK blocks
 * Block: THREAD_PER_BLOCK threads
 *
 * @tparam THREAD_PER_BLOCK Number of threads per block
 * @tparam ROW_PER_BLOCK Number of rows processed per block
 * @param input Input vector [N]
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output vector [M]
 * @param M Number of output rows
 * @param N Number of input columns
 */
template <i32 THREAD_PER_BLOCK, i32 ROW_PER_BLOCK>
__global__ void matmul_gemv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 M,
    i32 N);

/**
 * @brief CUDA kernel for matrix-matrix multiplication (GEMM)
 *
 * Computes: output[B × M] = input[B × N] @ weight[M × N]^T
 *
 * Each block computes multiple output elements.
 *
 * Grid:  (B × M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK blocks
 * Block: THREAD_PER_BLOCK threads
 *
 * @tparam THREAD_PER_BLOCK Number of threads per block
 * @tparam ROW_PER_BLOCK Number of rows processed per block
 * @param input Input matrix [B × N] (row-major)
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output matrix [B × M] (row-major)
 * @param B Batch size
 * @param M Output dimension
 * @param N Input dimension
 */
template <i32 THREAD_PER_BLOCK, i32 ROW_PER_BLOCK>
__global__ void matmul_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 B,
    i32 M,
    i32 N);

// ============================================================================
// Host-side Launch Functions
// ============================================================================

/**
 * @brief Launch MatMul CUDA kernel (GEMV: matrix-vector)
 *
 * @param input Input data pointer (device) [N]
 * @param weight Weight data pointer (device) [M × N]
 * @param output Output data pointer (device) [M]
 * @param M Output dimension
 * @param N Input dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 M,
    i32 N,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch MatMul CUDA kernel (GEMM: matrix-matrix)
 *
 * @param input Input data pointer (device) [B × N]
 * @param weight Weight data pointer (device) [M × N]
 * @param output Output data pointer (device) [B × M]
 * @param B Batch size
 * @param M Output dimension
 * @param N Input dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 B,
    i32 M,
    i32 N,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH
