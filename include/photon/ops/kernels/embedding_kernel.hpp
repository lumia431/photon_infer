/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file embedding_kernel.hpp
 * @brief CPU kernels for embedding operations
 * @version 0.1.0
 */


#include <span>

#ifdef PHOTON_USE_EIGEN
#include <Eigen/Core>
#endif

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels {

// ============================================================================
// Embedding CPU Kernels
// ============================================================================

#ifdef PHOTON_USE_EIGEN
/**
 * @brief CPU kernel for embedding lookup with bounds checking
 *
 * This function performs a batched embedding lookup:
 * - For each token ID in `tokens`, lookup the corresponding row in `weight`
 * - Copy the embedding vector to the output
 *
 * Memory layout:
 * - tokens: [batch_size] or [seq_len]
 * - weight: [vocab_size × embedding_dim] (row-major)
 * - output: [batch_size × embedding_dim] or [seq_len × embedding_dim] (row-major)
 *
 * Implementation details:
 * - Uses std::span for safe array access
 * - Eigen::Map for vectorized copy (SIMD optimized)
 * - Bounds checking for token indices
 *
 * @tparam TokenType Integer type for token IDs (i32, i64, etc.)
 * @tparam EmbedType Floating point type for embeddings (f32, f64)
 *
 * @param tokens Input token IDs
 * @param weight Embedding weight table (row-major)
 * @param output Output embedding vectors (row-major)
 * @param vocab_size Vocabulary size (for bounds checking)
 * @param embedding_dim Embedding dimension
 * @return Result indicating success or error
 */
template <std::integral TokenType, FloatingPoint EmbedType>
Result<void> embedding_forward_cpu(
    std::span<const TokenType> tokens,
    std::span<const EmbedType> weight,
    std::span<EmbedType> output,
    i32 vocab_size,
    i32 embedding_dim) noexcept {

  const usize num_tokens = tokens.size();

  // Validate input sizes
  if (weight.size() != static_cast<usize>(vocab_size * embedding_dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch");
  }

  if (output.size() != num_tokens * static_cast<usize>(embedding_dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch");
  }

  // Perform embedding lookup for each token
  for (usize i = 0; i < num_tokens; ++i) {
    const TokenType token = tokens[i];

    // Bounds check
    if (token < 0 || token >= vocab_size) {
      return Err<void>(
          ErrorCode::InvalidIndex,
          "Token index out of bounds: " + std::to_string(token) +
              " (vocab_size=" + std::to_string(vocab_size) + ")");
    }

    // Calculate pointers
    const auto* src = weight.data() + static_cast<usize>(token) * embedding_dim;
    auto* dst = output.data() + i * embedding_dim;

    // Use Eigen for vectorized copy (SIMD optimized)
    Eigen::Map<const Eigen::Matrix<EmbedType, Eigen::Dynamic, 1>> src_vec(
        src, embedding_dim);
    Eigen::Map<Eigen::Matrix<EmbedType, Eigen::Dynamic, 1>> dst_vec(
        dst, embedding_dim);

    dst_vec = src_vec;
  }

  return Ok();
}
#endif

/**
 * @brief Optimized batch embedding lookup (no bounds checking)
 *
 * This is a faster version that assumes token indices are already validated.
 * Use only when you're certain indices are within bounds.
 *
 * @param tokens Input token IDs (must be validated)
 * @param weight Embedding weight table
 * @param output Output embedding vectors
 * @param embedding_dim Embedding dimension
 */
template <std::integral TokenType, FloatingPoint EmbedType>
void embedding_forward_cpu_unchecked(
    std::span<const TokenType> tokens,
    std::span<const EmbedType> weight,
    std::span<EmbedType> output,
    i32 embedding_dim) noexcept {

  const usize num_tokens = tokens.size();

  for (usize i = 0; i < num_tokens; ++i) {
    const usize token = static_cast<usize>(tokens[i]);
    const auto* src = weight.data() + token * embedding_dim;
    auto* dst = output.data() + i * embedding_dim;

#ifdef PHOTON_USE_EIGEN
    // Vectorized copy using Eigen
    Eigen::Map<const Eigen::Matrix<EmbedType, Eigen::Dynamic, 1>> src_vec(
        src, embedding_dim);
    Eigen::Map<Eigen::Matrix<EmbedType, Eigen::Dynamic, 1>> dst_vec(
        dst, embedding_dim);

    dst_vec = src_vec;
#else
    // Manual copy when Eigen is disabled
    for (i32 j = 0; j < embedding_dim; ++j) {
      dst[j] = src[j];
    }
#endif
  }
}

}  // namespace photon::kernels

