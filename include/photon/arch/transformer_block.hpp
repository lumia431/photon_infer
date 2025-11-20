/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file transformer_block.hpp
 * @brief Transformer block for LLaMA architecture
 * @version 0.1.0
 */


#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include "photon/core/quant.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/rope.hpp"
#include "photon/ops/mha.hpp"
#include "photon/ops/add.hpp"
#include "photon/ops/swiglu.hpp"
#include "photon/arch/config.hpp"  // Import TransformerConfig

namespace photon::model {

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
   * @brief Forward pass through transformer block (single sequence)
   *
   * @param x Input/output tensor [dim] (modified in-place)
   * @param pos Current position in sequence
   * @param key_cache KV cache for keys [seq_len × kv_dim]
   * @param value_cache KV cache for values [seq_len × kv_dim]
   * @return Result<void> Success or error
   */
  Result<void> forward(Tensor& x, i32 pos, Tensor& key_cache, Tensor& value_cache);

  /**
   * @brief Batched forward pass through transformer block
   *
   * Processes multiple sequences in parallel using batch-optimized kernels
   * with block-based PagedAttention. This is the key optimization for achieving
   * high throughput with memory efficiency.
   *
   * @param x_batch Input/output tensor [total_tokens, dim] (modified in-place)
   * @param positions_gpu Positions on GPU [total_tokens]
   * @param positions_cpu Positions on CPU [total_tokens]
   * @param seq_ids_cpu Sequence IDs on CPU [total_tokens]
   * @param total_tokens Total number of tokens across all sequences in batch
   * @param key_cache Block-based key cache [num_blocks, num_kv_heads, block_size, head_size]
   * @param value_cache Block-based value cache [num_blocks, num_kv_heads, block_size, head_size]
   * @param cache_manager Paged cache manager with block table
   * @return Result<void> Success or error
   */
  Result<void> forward_batched(
      Tensor& x_batch,
      const i32* positions_gpu,
      const std::vector<i32>& positions_cpu,
      const std::vector<i32>& seq_ids_cpu,
      i32 total_tokens,
      Tensor& key_cache,
      Tensor& value_cache,
      class KVCacheManager* cache_manager);

  /**
   * @brief Quantize all MatMul weights in this block to INT8
   *
   * @param group_size Group size for quantization (default 128)
   * @return Result<void> Success or error
   */
  Result<void> quantize_weights(i32 group_size = 128);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief Set cuBLAS handle for all MatMul operators in this block
   *
   * @param handle cuBLAS handle for FP16 Tensor Core optimization
   */
  void set_cublas_handle(void* handle) {
    wq_.set_cublas_handle(handle);
    wk_.set_cublas_handle(handle);
    wv_.set_cublas_handle(handle);
    wo_.set_cublas_handle(handle);
    w1_.set_cublas_handle(handle);
    w2_.set_cublas_handle(handle);
    w3_.set_cublas_handle(handle);
  }
#endif

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

  // Intermediate buffers (single-sequence)
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

  // Batched intermediate buffers (allocated on-demand)
  i32 current_batch_capacity_ = 0;  // Track allocated capacity in total_tokens
  Tensor attn_out_batch_;     // [total_tokens, dim]
  Tensor q_batch_;            // [total_tokens, dim]
  Tensor k_batch_;            // [total_tokens, kv_dim]
  Tensor v_batch_;            // [total_tokens, kv_dim]
  Tensor attn_result_batch_;  // [total_tokens, dim]
  Tensor wo_out_batch_;       // [total_tokens, dim]
  Tensor ffn_out_batch_;      // [total_tokens, dim]
  Tensor w1_out_batch_;       // [total_tokens, hidden_dim]
  Tensor w3_out_batch_;       // [total_tokens, hidden_dim]
  Tensor swiglu_out_batch_;   // [total_tokens, hidden_dim]
  Tensor w2_out_batch_;       // [total_tokens, dim]

  // Temporary cache buffers for MHA (reused across forward calls)
  Tensor temp_key_cache_;     // [seq_len, kv_dim] - reusable buffer
  Tensor temp_value_cache_;   // [seq_len, kv_dim] - reusable buffer

  // Cached GPU buffers for batched MHA (reused across forward calls)
  Tensor cached_offsets_gpu_;  // [total_tokens] - cache offsets on GPU
  Tensor cached_score_buf_;    // [total_tokens, n_heads, seq_len] - score buffer

  // Cache validation for offset caching
  std::vector<i32> cached_seq_ids_;  // Track which seq_ids we've cached offsets for
  bool offsets_valid_ = false;

  bool initialized_ = false;

  /**
   * @brief Ensure batched buffers are allocated for given total_tokens
   */
  Result<void> ensure_batch_buffers(i32 total_tokens);
};

}  // namespace photon::model

