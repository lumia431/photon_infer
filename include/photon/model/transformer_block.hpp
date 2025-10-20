/**
 * @file transformer_block.hpp
 * @brief Transformer block for LLaMA architecture
 * @version 0.1.0
 */

#ifndef PHOTON_MODEL_TRANSFORMER_BLOCK_HPP
#define PHOTON_MODEL_TRANSFORMER_BLOCK_HPP

#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/rope.hpp"
#include "photon/ops/mha.hpp"
#include "photon/ops/add.hpp"
#include "photon/ops/swiglu.hpp"

namespace photon::model {

/**
 * @struct TransformerConfig
 * @brief Configuration for LLaMA transformer model
 */
struct TransformerConfig {
  i32 vocab_size;    // Vocabulary size
  i32 dim;           // Model dimension
  i32 hidden_dim;    // FFN hidden dimension
  i32 n_layers;      // Number of transformer blocks
  i32 n_heads;       // Number of attention heads
  i32 n_kv_heads;    // Number of KV heads (for GQA)
  i32 head_size;     // Dimension per head
  i32 seq_len;       // Maximum sequence length
  f32 norm_eps;      // RMSNorm epsilon

  // Computed values
  i32 kv_dim;        // KV dimension (n_kv_heads * head_size)
  i32 kv_mul;        // Heads per KV head (n_heads / n_kv_heads)

  TransformerConfig() = default;

  void compute_derived() {
    kv_dim = n_kv_heads * head_size;
    kv_mul = n_heads / n_kv_heads;
  }
};

/**
 * @class TransformerBlock
 * @brief Single transformer layer implementation
 *
 * Architecture:
 * ```
 * x = embedding(tokens)
 * for layer in layers:
 *     # Self-attention
 *     h = rmsnorm(x)
 *     q, k, v = wq(h), wk(h), wv(h)
 *     q, k = rope(q, k, pos)
 *     attn_out = mha(q, k_cache, v_cache)
 *     x = x + wo(attn_out)   # Residual
 *
 *     # Feed-forward
 *     h = rmsnorm(x)
 *     ffn_out = w2(swiglu(w1(h), w3(h)))
 *     x = x + ffn_out        # Residual
 *
 * logits = rmsnorm(x) @ wcls
 * ```
 */
class TransformerBlock {
 public:
  /**
   * @brief Construct transformer block
   *
   * @param layer_idx Layer index (0-based)
   * @param config Model configuration
   */
  explicit TransformerBlock(i32 layer_idx, const TransformerConfig& config);

  /**
   * @brief Initialize the block (must be called before use)
   */
  Result<void> init();

  /**
   * @brief Set attention weight matrices
   */
  Result<void> set_wq(Tensor weight);
  Result<void> set_wk(Tensor weight);
  Result<void> set_wv(Tensor weight);
  Result<void> set_wo(Tensor weight);

  /**
   * @brief Set FFN weight matrices
   */
  Result<void> set_w1(Tensor weight);
  Result<void> set_w2(Tensor weight);
  Result<void> set_w3(Tensor weight);

  /**
   * @brief Set RMSNorm weights
   */
  Result<void> set_attn_norm(Tensor weight);
  Result<void> set_ffn_norm(Tensor weight);

  /**
   * @brief Forward pass through transformer block
   *
   * @param x Input/output tensor [dim] (modified in-place)
   * @param pos Current position in sequence
   * @param key_cache KV cache for keys [seq_len × kv_dim]
   * @param value_cache KV cache for values [seq_len × kv_dim]
   * @return Result<void> Success or error
   */
  Result<void> forward(Tensor& x, i32 pos, Tensor& key_cache, Tensor& value_cache);

  [[nodiscard]] constexpr i32 layer_idx() const noexcept { return layer_idx_; }

 private:
  i32 layer_idx_;
  TransformerConfig config_;

  // Operators
  RMSNormOp attn_norm_;
  RMSNormOp ffn_norm_;
  MatMulOp wq_;
  MatMulOp wk_;
  MatMulOp wv_;
  MatMulOp wo_;
  MatMulOp w1_;
  MatMulOp w2_;
  MatMulOp w3_;
  RoPEOp rope_;
  MHAOp mha_;
  AddOp add_;
  SwiGLUOp swiglu_;

  // Intermediate buffers
  Tensor attn_out_;     // After attention normalization
  Tensor q_;            // Query [dim]
  Tensor k_;            // Key [kv_dim]
  Tensor v_;            // Value [kv_dim]
  Tensor attn_result_;  // MHA output [dim]
  Tensor wo_out_;       // After wo projection [dim]
  Tensor ffn_out_;      // After FFN normalization [dim]
  Tensor w1_out_;       // After w1 [hidden_dim]
  Tensor w3_out_;       // After w3 [hidden_dim]
  Tensor swiglu_out_;   // After SwiGLU [hidden_dim]
  Tensor w2_out_;       // After w2 [dim]

  bool initialized_ = false;
};

}  // namespace photon::model

#endif  // PHOTON_MODEL_TRANSFORMER_BLOCK_HPP
