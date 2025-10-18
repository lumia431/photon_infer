/**
 * @file embedding.hpp
 * @brief Embedding layer operator
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_EMBEDDING_HPP
#define PHOTON_OPS_EMBEDDING_HPP

#include <span>
#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// Embedding Operator
// ============================================================================

/**
 * @class EmbeddingOp
 * @brief Embedding layer for token-to-vector lookup
 *
 * This operator implements a simple embedding lookup table that maps
 * integer token IDs to dense vector representations.
 *
 * Architecture:
 * - Input: Token IDs [batch_size] or [seq_len] (int32)
 * - Weight: Embedding table [vocab_size × embedding_dim] (float32)
 * - Output: Embeddings [batch_size × embedding_dim] or [seq_len × embedding_dim]
 *
 * Example:
 * ```cpp
 * // vocab_size=1000, embedding_dim=256
 * EmbeddingOp op(1000, 256);
 *
 * // Set pretrained weights
 * auto weight = Tensor::create({1000, 256}, DataType::Float32).value();
 * op.set_weight(std::move(weight));
 * op.init();
 *
 * // Forward pass: [3 token IDs] -> [3 × 256 embeddings]
 * auto tokens = Tensor::from_vector<i32>({100, 205, 789}).value();
 * auto output = Tensor::create({3, 256}, DataType::Float32).value();
 * op.forward(tokens, output);
 * ```
 *
 * Implementation:
 * - Uses CRTP pattern from OperatorBase
 * - Zero-copy weight access via std::span
 * - Vectorized copy via Eigen::Map
 * - Bounds checking for token indices
 */
class EmbeddingOp : public ParameterizedOperator<EmbeddingOp> {
 public:
  /**
   * @brief Construct embedding operator
   *
   * @param vocab_size Number of tokens in vocabulary
   * @param embedding_dim Dimension of embedding vectors
   */
  explicit EmbeddingOp(i32 vocab_size, i32 embedding_dim)
      : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {
    // Pre-allocate weight storage
    weights_.resize(1);
  }

  /**
   * @brief Set embedding weight matrix
   *
   * @param weight Tensor of shape [vocab_size × embedding_dim]
   * @return Result indicating success or error
   */
  Result<void> set_weight(Tensor weight) {
    // Validate weight shape
    if (weight.ndim() != 2) {
      return Err<void>(ErrorCode::InvalidShape,
                      "Embedding weight must be 2D tensor");
    }

    if (weight.dim(0) != vocab_size_ || weight.dim(1) != embedding_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Weight shape mismatch: expected [" + std::to_string(vocab_size_) +
              " × " + std::to_string(embedding_dim_) + "], got [" +
              std::to_string(weight.dim(0)) + " × " +
              std::to_string(weight.dim(1)) + "]");
    }

    if (weight.dtype() != DataType::Float32) {
      return Err<void>(ErrorCode::InvalidDtype,
                      "Embedding weight must be Float32");
    }

    if (weight.device() != device_) {
      return Err<void>(ErrorCode::DeviceMismatch,
                      "Weight device does not match operator device");
    }

    weights_[0] = std::move(weight);
    return Ok();
  }

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    if (!weights_initialized()) {
      return Err<void>(ErrorCode::InvalidOperator,
                      "Embedding weights not set");
    }
    return Ok();
  }

  /**
   * @brief Forward pass: token IDs -> embeddings
   *
   * @param input Token IDs, shape [batch_size] or [seq_len], dtype Int32
   * @param output Embeddings, shape [batch_size × embedding_dim] or [seq_len × embedding_dim], dtype Float32
   * @return Result indicating success or error
   */
  Result<void> forward(const Tensor& input, Tensor& output);

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "EmbeddingOp";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Embedding;
  }

  /**
   * @brief Get vocab size
   */
  [[nodiscard]] i32 vocab_size() const noexcept { return vocab_size_; }

  /**
   * @brief Get embedding dimension
   */
  [[nodiscard]] i32 embedding_dim() const noexcept { return embedding_dim_; }

 private:
  i32 vocab_size_;      ///< Vocabulary size
  i32 embedding_dim_;   ///< Embedding vector dimension

  /**
   * @brief CPU forward implementation
   */
  Result<void> forward_cpu(const Tensor& input, Tensor& output);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief CUDA forward implementation
   */
  Result<void> forward_cuda(const Tensor& input, Tensor& output);
#endif
};

// Verify EmbeddingOp satisfies Operator concept
static_assert(Operator<EmbeddingOp>, "EmbeddingOp must satisfy Operator concept");
static_assert(UnaryOperator<EmbeddingOp>, "EmbeddingOp must satisfy UnaryOperator concept");

}  // namespace photon

#endif  // PHOTON_OPS_EMBEDDING_HPP
