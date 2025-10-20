#ifndef PHOTON_OPS_MHA_HPP
#define PHOTON_OPS_MHA_HPP

#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

/**
 * @brief Multi-Head Attention operator
 *
 * Implements the core computation of multi-head attention mechanism:
 * 1. For each head h and each position t <= pos:
 *    score[h][t] = (Q[h] · K[h][t]) / sqrt(head_size)
 * 2. Apply softmax to scores: attn[h] = softmax(score[h][0:pos+1])
 * 3. Aggregate values: output[h] = sum(attn[h][t] * V[h][t] for t in 0:pos+1)
 *
 * This operator supports Grouped Query Attention (GQA) where the number of
 * key/value heads can be different from query heads (controlled by kv_mul).
 *
 * @tparam T The floating point type (f32 or f64)
 */
class MHAOp : public OperatorBase<MHAOp> {
 public:
  /**
   * @brief Construct Multi-Head Attention operator
   *
   * @param dim Total query dimension (head_num * head_size)
   * @param kv_dim Total key/value dimension (kv_heads * head_size)
   * @param head_num Number of query heads
   * @param head_size Dimension of each head
   * @param seq_len Maximum sequence length
   * @param use_naive Use naive implementation instead of optimized
   */
  explicit MHAOp(i32 dim, i32 kv_dim, i32 head_num, i32 head_size,
                 i32 seq_len, bool use_naive = false);

  /**
   * @brief Forward pass of multi-head attention
   *
   * @param query Query tensor [dim] = [head_num × head_size]
   * @param key_cache Key cache tensor [seq_len × kv_dim]
   * @param value_cache Value cache tensor [seq_len × kv_dim]
   * @param output Output tensor [dim]
   * @param pos Current position in sequence (0 to seq_len-1)
   * @return Result<void> Success or error
   *
   * Intermediate computation:
   * - score: [head_num × seq_len] attention scores (allocated internally)
   * - For each head h: score[h][0:pos+1] stores attention weights
   */
  Result<void> forward(const Tensor& query, const Tensor& key_cache,
                      const Tensor& value_cache, Tensor& output, i32 pos);

  [[nodiscard]] constexpr i32 dim() const noexcept { return dim_; }
  [[nodiscard]] constexpr i32 kv_dim() const noexcept { return kv_dim_; }
  [[nodiscard]] constexpr i32 head_num() const noexcept { return head_num_; }
  [[nodiscard]] constexpr i32 head_size() const noexcept { return head_size_; }
  [[nodiscard]] constexpr i32 seq_len() const noexcept { return seq_len_; }
  [[nodiscard]] constexpr i32 kv_mul() const noexcept { return kv_mul_; }
  [[nodiscard]] constexpr bool use_naive() const noexcept { return use_naive_; }

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    // No special initialization needed for MHA
    return Ok();
  }

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "MHA";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Attention;
  }

 private:
  i32 dim_;        // Total query dimension (head_num * head_size)
  i32 kv_dim_;     // Total key/value dimension (for GQA)
  i32 head_num_;   // Number of query heads
  i32 head_size_;  // Dimension per head
  i32 seq_len_;    // Maximum sequence length
  i32 kv_mul_;     // Query heads per KV head (dim/kv_dim)
  bool use_naive_; // Use naive implementation
};

// Verify MHAOp satisfies the Operator concept
static_assert(Operator<MHAOp>, "MHAOp must satisfy Operator concept");

}  // namespace photon

#endif  // PHOTON_OPS_MHA_HPP
