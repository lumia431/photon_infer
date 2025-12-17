/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file llama_model.hpp
 * @brief LLaMA language model implementation
 * @version 0.1.0
 */


#include "photon/arch/transformer_block.hpp"
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
   * @brief Destructor (needed for unique_ptr with forward-declared type)
   */
  ~LLaMAModel();

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

  /**
   * @brief Quantize all model weights to INT8
   *
   * Quantizes all MatMul weights (transformer blocks + classifier) in-place.
   * This significantly reduces memory usage (~3.8x compression) with minimal
   * accuracy loss (RMSE < 0.06).
   *
   * Must be called after weights are loaded and before inference begins.
   *
   * @param group_size Group size for quantization (default 128)
   * @return Result<void> Success or error
   */
  Result<void> quantize_weights(i32 group_size = 128);

  /**
   * @brief Initialize block-based PagedAttention cache
   *
   * Uses KVCacheManager with BlockManager and BlockTable for true
   * non-contiguous block allocation. Provides better memory efficiency (50-80% savings).
   *
   * @param num_blocks Total number of physical blocks
   * @param block_size Tokens per block (typically 16)
   * @return Result<void> Success or error
   */
  Result<void> init_paged_cache(i32 num_blocks, i32 block_size);

  /**
   * @brief Get paged cache manager
   */
  class KVCacheManager* paged_cache_manager() { return paged_cache_manager_.get(); }

  /**
   * @brief Batched forward pass (multiple sequences in parallel)
   *
   * @param tokens Token IDs for each sequence [total_tokens]
   * @param positions Position for each sequence [total_tokens]
   * @param seq_ids Sequence IDs [total_tokens]
   * @param logits Output logits [total_tokens, vocab_size] (must be on GPU)
   * @return Result<void> Success or error
   */
  Result<void> forward_batched(
      const std::vector<i32>& tokens,
      const std::vector<i32>& positions,
      const std::vector<i32>& seq_ids,
      Tensor& logits);


  [[nodiscard]] const TransformerConfig& config() const noexcept { return config_; }

 private:
  TransformerConfig config_;

  // Model components
  EmbeddingOp embedding_;
  std::vector<std::unique_ptr<TransformerBlock>> blocks_;
  RMSNormOp final_norm_;
  MatMulOp classifier_;

  // KV cache: [n_layers][seq_len × kv_dim]
  std::vector<Tensor> key_cache_;    // One per layer (for single-sequence mode)
  std::vector<Tensor> value_cache_;  // One per layer (for single-sequence mode)

  // Block-based PagedAttention cache manager (for multi-sequence mode)
  std::unique_ptr<class KVCacheManager> paged_cache_manager_;

  // Working buffers
  Tensor x_;           // Current hidden state [dim]
  Tensor emb_out_;     // Embedding output [dim]
  Tensor norm_out_;    // Final norm output [dim]
  Tensor logits_buf_;  // Logits buffer [vocab_size]

  // Pre-allocated GPU buffers for batched inference (reused across calls)
  i32 batched_buffers_capacity_ = 0;  // Allocated capacity in total_tokens
  i32* positions_gpu_ = nullptr;      // [total_tokens] - reusable
  i32* seq_ids_gpu_ = nullptr;        // [total_tokens] - reusable
  i32* cache_offsets_gpu_ = nullptr;  // [total_tokens] - reusable
  Tensor tokens_batch_;                // [total_tokens] - reusable
  Tensor x_batch_;                     // [total_tokens, dim] - reusable
  Tensor norm_batch_;                  // [total_tokens, dim] - reusable

  bool initialized_ = false;
  bool use_paged_cache_ = false;   // Whether to use block-based PagedAttention cache

#ifdef PHOTON_USE_CUDA
  void* cublas_handle_ = nullptr;  // cuBLAS handle for FP16 Tensor Core optimization
#endif

  /**
   * @brief Ensure batched buffers are allocated for given total_tokens
   */
  Result<void> ensure_batched_buffers(i32 total_tokens);
};

/**
 * @brief Simple argmax sampler for token selection
 *
 * @param logits Logits tensor [vocab_size]
 * @return Result<i32> Token with highest logit
 */
Result<i32> argmax_sample(const Tensor& logits);

}  // namespace photon::model

