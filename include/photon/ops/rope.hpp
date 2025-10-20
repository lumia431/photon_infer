/**
 * @file rope.hpp
 * @brief Rotary Position Embedding (RoPE) operator
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_ROPE_HPP
#define PHOTON_OPS_ROPE_HPP

#include <vector>
#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// RoPE Operator
// ============================================================================

/**
 * @class RoPEOp
 * @brief Rotary Position Embedding for transformer models
 *
 * RoPE applies rotary embeddings to query and key vectors by rotating
 * pairs of elements based on their position in the sequence. This provides
 * relative positional information without absolute position encodings.
 *
 * Algorithm:
 * ```
 * For each pair (x[i], x[i+1]) in the embedding:
 *   head_dim = i % head_size
 *   freq = 1.0 / pow(10000, head_dim / head_size)
 *   angle = pos * freq
 *
 *   // 2D rotation matrix
 *   x[i]   = x[i] * cos(angle) - x[i+1] * sin(angle)
 *   x[i+1] = x[i] * sin(angle) + x[i+1] * cos(angle)
 * ```
 *
 * Architecture:
 * - Input Q: [dim] - Query vector
 * - Input K: [kv_dim] - Key vector (can be different for GQA/MQA)
 * - Input Pos: scalar integer - Current position in sequence
 * - Output: Modified Q and K tensors (in-place)
 *
 * Parameters:
 * - dim: Total dimension of query (e.g., 2048)
 * - kv_dim: Dimension of key/value (same as dim for MHA, smaller for GQA)
 * - head_size: Dimension per attention head (e.g., 128)
 * - max_seq_len: Maximum sequence length for precomputed cache
 *
 * Example:
 * ```cpp
 * // LLaMA 3.2-1B: 16 heads × 64 dim = 1024 total
 * // GQA: 4 KV heads × 64 dim = 256 kv_dim
 * RoPEOp rope(1024, 256, 64, 2048);
 * rope.init();
 *
 * // At position 5 in sequence
 * auto q = Tensor::create({1024}, DataType::Float32).value();
 * auto k = Tensor::create({256}, DataType::Float32).value();
 * rope.forward(q, k, 5);  // Applies rotation in-place
 * ```
 *
 * Features:
 * - Precomputed sin/cos cache for all positions
 * - Supports GQA (Grouped Query Attention) with different Q/K dimensions
 * - In-place modification for memory efficiency
 * - Naive and Eigen-optimized implementations
 */
class RoPEOp : public OperatorBase<RoPEOp> {
 public:
  /**
   * @brief Construct RoPE operator
   *
   * @param dim Query dimension (total across all heads)
   * @param kv_dim Key/Value dimension (for GQA, same as dim for MHA)
   * @param head_size Dimension per attention head
   * @param max_seq_len Maximum sequence length
   * @param use_naive Use naive implementation (for benchmarking)
   */
  explicit RoPEOp(i32 dim, i32 kv_dim, i32 head_size, i32 max_seq_len,
                  bool use_naive = false)
      : dim_(dim),
        kv_dim_(kv_dim),
        head_size_(head_size),
        max_seq_len_(max_seq_len),
        use_naive_(use_naive) {}

  /**
   * @brief Initialize the operator and precompute sin/cos cache
   */
  Result<void> init_impl();

  /**
   * @brief Forward pass: apply rotary embeddings
   *
   * Applies rotation to query and key vectors based on position.
   * Modifies tensors in-place for efficiency.
   *
   * @param q Query tensor [dim], will be modified in-place
   * @param k Key tensor [kv_dim], will be modified in-place
   * @param pos Position in sequence (0-indexed)
   * @return Result indicating success or error
   */
  Result<void> forward(Tensor& q, Tensor& k, i32 pos);

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "RoPEOp";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Attention;
  }

  /**
   * @brief Get query dimension
   */
  [[nodiscard]] i32 dim() const noexcept { return dim_; }

  /**
   * @brief Get key/value dimension
   */
  [[nodiscard]] i32 kv_dim() const noexcept { return kv_dim_; }

  /**
   * @brief Get head size
   */
  [[nodiscard]] i32 head_size() const noexcept { return head_size_; }

  /**
   * @brief Get maximum sequence length
   */
  [[nodiscard]] i32 max_seq_len() const noexcept { return max_seq_len_; }

  /**
   * @brief Check if using naive implementation
   */
  [[nodiscard]] bool is_naive() const noexcept { return use_naive_; }

  /**
   * @brief Get sin cache (for testing)
   */
  [[nodiscard]] const std::vector<f32>& sin_cache() const noexcept {
    return sin_cache_;
  }

  /**
   * @brief Get cos cache (for testing)
   */
  [[nodiscard]] const std::vector<f32>& cos_cache() const noexcept {
    return cos_cache_;
  }

 private:
  i32 dim_;          ///< Query dimension
  i32 kv_dim_;       ///< Key/Value dimension
  i32 head_size_;    ///< Dimension per head
  i32 max_seq_len_;  ///< Maximum sequence length
  bool use_naive_;   ///< Use naive implementation flag

  // Precomputed sin/cos cache: [max_seq_len × head_size]
  std::vector<f32> sin_cache_;
  std::vector<f32> cos_cache_;

  /**
   * @brief CPU forward implementation
   */
  Result<void> forward_cpu(Tensor& q, Tensor& k, i32 pos);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief CUDA forward implementation
   */
  Result<void> forward_cuda(Tensor& q, Tensor& k, i32 pos);
#endif
};

// Verify RoPEOp satisfies Operator concept
static_assert(Operator<RoPEOp>, "RoPEOp must satisfy Operator concept");

}  // namespace photon

#endif  // PHOTON_OPS_ROPE_HPP
