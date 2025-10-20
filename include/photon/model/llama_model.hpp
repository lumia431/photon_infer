/**
 * @file llama_model.hpp
 * @brief LLaMA language model implementation
 * @version 0.1.0
 */

#ifndef PHOTON_MODEL_LLAMA_MODEL_HPP
#define PHOTON_MODEL_LLAMA_MODEL_HPP

#include "photon/model/transformer_block.hpp"
#include "photon/ops/embedding.hpp"
#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/core/tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace photon::model {

/**
 * @class LLaMAModel
 * @brief Complete LLaMA language model
 *
 * Architecture:
 * ```
 * 1. Token Embedding
 * 2. N Transformer Blocks (with KV cache)
 * 3. Final RMSNorm
 * 4. Classifier (lm_head)
 * ```
 *
 * Features:
 * - KV cache for efficient autoregressive generation
 * - Supports prompt prefill and token-by-token generation
 * - Configurable model size (1B, 3B, 7B, etc.)
 *
 * Example usage:
 * ```cpp
 * TransformerConfig config;
 * config.vocab_size = 32000;
 * config.dim = 2048;
 * // ... set other params
 * config.compute_derived();
 *
 * LLaMAModel model(config);
 * model.init();
 * // Load weights...
 *
 * // Generate text
 * std::vector<i32> tokens = {1, 123, 456};  // Prompt tokens
 * auto next_token = model.generate_next(tokens);
 * ```
 */
class LLaMAModel {
 public:
  /**
   * @brief Construct LLaMA model
   *
   * @param config Model configuration
   */
  explicit LLaMAModel(const TransformerConfig& config);

  /**
   * @brief Initialize model (allocate buffers, init operators)
   */
  Result<void> init();

  /**
   * @brief Set embedding table
   *
   * @param weight Embedding weights [vocab_size × dim]
   */
  Result<void> set_embedding(Tensor weight);

  /**
   * @brief Set final normalization weight
   *
   * @param weight RMSNorm weight [dim]
   */
  Result<void> set_final_norm(Tensor weight);

  /**
   * @brief Set classifier (lm_head) weight
   *
   * @param weight Classifier weight [vocab_size × dim]
   */
  Result<void> set_classifier(Tensor weight);

  /**
   * @brief Get transformer block for weight loading
   *
   * @param layer_idx Layer index (0-based)
   * @return Reference to transformer block
   */
  TransformerBlock& get_block(i32 layer_idx);

  /**
   * @brief Forward pass (single token or prompt)
   *
   * @param token Token ID
   * @param pos Position in sequence
   * @param logits Output logits [vocab_size]
   * @return Result<void> Success or error
   *
   * This performs:
   * 1. Embedding lookup
   * 2. Forward through all transformer blocks
   * 3. Final normalization
   * 4. Classifier projection to get logits
   */
  Result<void> forward(i32 token, i32 pos, Tensor& logits);

  /**
   * @brief Generate next token (argmax sampling)
   *
   * @param tokens Input token sequence
   * @return Result<i32> Next token ID or error
   *
   * For prompt tokens, processes all tokens sequentially to fill KV cache.
   * Then returns the argmax of final logits.
   */
  Result<i32> generate_next(const std::vector<i32>& tokens);

  /**
   * @brief Reset KV cache (clear all cached keys/values)
   */
  void reset_cache();

  [[nodiscard]] const TransformerConfig& config() const noexcept { return config_; }

 private:
  TransformerConfig config_;

  // Model components
  EmbeddingOp embedding_;
  std::vector<std::unique_ptr<TransformerBlock>> blocks_;
  RMSNormOp final_norm_;
  MatMulOp classifier_;

  // KV cache: [n_layers][seq_len × kv_dim]
  std::vector<Tensor> key_cache_;    // One per layer
  std::vector<Tensor> value_cache_;  // One per layer

  // Working buffers
  Tensor x_;           // Current hidden state [dim]
  Tensor emb_out_;     // Embedding output [dim]
  Tensor norm_out_;    // Final norm output [dim]
  Tensor logits_buf_;  // Logits buffer [vocab_size]

  bool initialized_ = false;
};

/**
 * @brief Simple argmax sampler for token selection
 *
 * @param logits Logits tensor [vocab_size]
 * @return Result<i32> Token with highest logit
 */
Result<i32> argmax_sample(const Tensor& logits);

}  // namespace photon::model

#endif  // PHOTON_MODEL_LLAMA_MODEL_HPP
