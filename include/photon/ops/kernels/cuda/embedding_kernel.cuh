/**
 * @file embedding_kernel.cuh
 * @brief CUDA kernels for embedding lookup
 * @version 0.1.0
 *
 * Optimized embedding lookup with:
 * - Vectorized memory access (float4)
 * - Bounds checking
 * - Efficient token-to-embedding mapping
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_EMBEDDING_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_EMBEDDING_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Configuration
// ============================================================================

/// Block size for embedding lookup
inline constexpr i32 EMBEDDING_BLOCK_SIZE = 128;

/// Maximum sequence length (grid size limit)
inline constexpr i32 EMBEDDING_MAX_SEQ_LEN = 512;

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

/**
 * @brief CUDA kernel for embedding lookup
 *
 * Each block processes one token:
 * - blockIdx.x: token index
 * - threads: copy embedding_dim elements
 *
 * Grid:  num_tokens blocks
 * Block: EMBEDDING_BLOCK_SIZE threads
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param tokens Input token IDs [num_tokens]
 * @param weight Embedding table [vocab_size × embedding_dim]
 * @param output Output embeddings [num_tokens × embedding_dim]
 * @param num_tokens Number of tokens
 * @param vocab_size Vocabulary size
 * @param embedding_dim Embedding dimension
 */
template <i32 BLOCK_SIZE>
__global__ void embedding_kernel(
    const i32* __restrict__ tokens,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim);

// ============================================================================
// Host-side Launch Function
// ============================================================================

/**
 * @brief Launch embedding CUDA kernel
 *
 * @param tokens Token IDs (device) [num_tokens]
 * @param weight Embedding weights (device) [vocab_size × embedding_dim]
 * @param output Output embeddings (device) [num_tokens × embedding_dim]
 * @param num_tokens Number of tokens
 * @param vocab_size Vocabulary size
 * @param embedding_dim Embedding dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> embedding_cuda_launch(
    std::span<const i32> tokens,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_EMBEDDING_KERNEL_CUH
