/**
 * @file cuda_kernels.hpp
 * @brief CUDA kernel launch interfaces (C++ compatible)
 * @version 0.1.0
 *
 * This file provides C++-compatible function declarations for CUDA kernels.
 * It can be safely included in .cpp files without requiring CUDA compiler.
 *
 * Design rationale:
 * - .cpp files are compiled by C++ compiler (g++)
 * - .cu files are compiled by CUDA compiler (nvcc)
 * - .cuh files contain __global__ keywords which C++ compiler doesn't understand
 * - This .hpp file provides forward declarations that both compilers can understand
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_KERNELS_HPP
#define PHOTON_OPS_KERNELS_CUDA_KERNELS_HPP

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>

// Forward declare CUDA types to avoid requiring cuda_runtime.h
struct CUstream_st;
using cudaStream_t = struct CUstream_st*;

namespace photon::kernels::cuda {

// ============================================================================
// RMSNorm CUDA Kernels
// ============================================================================

/**
 * @brief Launch RMSNorm CUDA kernel (single vector)
 *
 * @param input Input data pointer (device) [dim]
 * @param weight Weight data pointer (device) [dim]
 * @param output Output data pointer (device) [dim]
 * @param dim Feature dimension
 * @param eps Epsilon value
 * @param stream CUDA stream (nullptr = default stream)
 * @return Result indicating success or error
 */
Result<void> rmsnorm_cuda_launch(
    const f32* input,
    const f32* weight,
    f32* output,
    i32 dim,
    f32 eps,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch RMSNorm CUDA kernel (batch)
 *
 * @param input Input data pointer (device) [batch_size × dim]
 * @param weight Weight data pointer (device) [dim]
 * @param output Output data pointer (device) [batch_size × dim]
 * @param batch_size Number of vectors
 * @param dim Feature dimension
 * @param eps Epsilon value
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> rmsnorm_batch_cuda_launch(
    const f32* input,
    const f32* weight,
    f32* output,
    i32 batch_size,
    i32 dim,
    f32 eps,
    cudaStream_t stream = nullptr);

// ============================================================================
// MatMul CUDA Kernels
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

// ============================================================================
// Add CUDA Kernel
// ============================================================================

/**
 * @brief Launch element-wise addition CUDA kernel
 *
 * @param input1 First input data (device) [size]
 * @param input2 Second input data (device) [size]
 * @param output Output data (device) [size]
 * @param size Number of elements
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> add_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream = nullptr);

// ============================================================================
// RoPE CUDA Kernels
// ============================================================================

/**
 * @brief Launch RoPE cache computation CUDA kernel
 *
 * Pre-computes sin/cos values for all positions and dimensions.
 *
 * @param sin_cache Sin cache (device) [max_seq_len × head_size]
 * @param cos_cache Cos cache (device) [max_seq_len × head_size]
 * @param max_seq_len Maximum sequence length
 * @param head_size Head dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> compute_rope_cache_cuda_launch(
    std::span<f32> sin_cache,
    std::span<f32> cos_cache,
    i32 max_seq_len,
    i32 head_size,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch RoPE CUDA kernel
 *
 * Applies rotary position embedding to query and key tensors.
 *
 * @param query Query tensor (device) [dim]
 * @param key Key tensor (device) [kv_dim]
 * @param sin_cache Sin cache (device) [max_seq_len × head_size]
 * @param cos_cache Cos cache (device) [max_seq_len × head_size]
 * @param pos Current position
 * @param dim Query dimension
 * @param kv_dim Key dimension
 * @param head_size Head dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> rope_cuda_launch(
    std::span<f32> query,
    std::span<f32> key,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream = nullptr);

// ============================================================================
// MHA CUDA Kernel
// ============================================================================

/**
 * @brief Launch Multi-Head Attention CUDA kernel
 *
 * Computes attention for all heads in parallel using the KV cache.
 *
 * @param query Query tensor (device) [dim = head_num × head_size]
 * @param key_cache Key cache (device) [layer_num × seq_len × kv_dim]
 * @param value_cache Value cache (device) [layer_num × seq_len × kv_dim]
 * @param output Output tensor (device) [dim]
 * @param score Scratch buffer (device) [head_num × seq_len]
 * @param pos Current position (attend to [0:pos])
 * @param layer_idx Current layer index
 * @param seq_len Maximum sequence length
 * @param kv_dim Key/Value dimension
 * @param head_num Number of query heads
 * @param head_size Dimension per head
 * @param kv_mul Query heads per KV head (for GQA)
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> mha_cuda_launch(
    std::span<const f32> query,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    std::span<f32> output,
    std::span<f32> score,
    i32 pos,
    i32 layer_idx,
    i32 seq_len,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 kv_mul,
    cudaStream_t stream = nullptr);

// ============================================================================
// SwiGLU CUDA Kernel
// ============================================================================

/**
 * @brief Launch SwiGLU CUDA kernel
 *
 * Computes: output = Swish(input1) * input2
 * where Swish(x) = x * sigmoid(x)
 *
 * @param input1 First input tensor (device) [size]
 * @param input2 Second input tensor (device) [size]
 * @param output Output tensor (device) [size]
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

// ============================================================================
// Embedding CUDA Kernel
// ============================================================================

/**
 * @brief Launch Embedding CUDA kernel
 *
 * Looks up embeddings for a batch of tokens.
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

#endif  // PHOTON_OPS_KERNELS_CUDA_KERNELS_HPP
