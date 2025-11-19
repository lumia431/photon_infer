/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file kv_cache_manager.hpp
 * @brief Block-based KV Cache Manager with PagedAttention support
 * @version 2.0.0
 *
 * Implements true block-based paged attention with dynamic block allocation.
 */

#include "photon/core/tensor.hpp"
#include "photon/core/error.hpp"
#include "photon/runtime/block_manager.hpp"
#include "photon/runtime/block_table.hpp"
#include <vector>
#include <memory>

namespace photon::model {

/**
 * @brief Block-based KV Cache Manager for PagedAttention
 *
 * **Design:**
 * - Dynamic block allocation: allocate blocks on-demand
 * - Block table: non-contiguous memory access via block table
 * - Memory efficient: only allocate what's needed (50-80% savings)
 * - Dynamic extension: sequences can grow beyond initial allocation
 *
 * **Memory Layout:**
 * - KV Cache: [num_blocks, num_kv_heads, block_size, head_size]
 * - Each sequence maps to multiple blocks via BlockTable
 * - Blocks are non-contiguous and managed by BlockManager
 *
 * **Example:**
 * ```
 * Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7]...
 * Seq 0: [B0, B3, B7]   <- 3 blocks, non-contiguous
 * Seq 1: [B1, B4]       <- 2 blocks
 * Seq 2: [B2, B5, B6]   <- 3 blocks
 * ```
 */
class KVCacheManager {
 public:
  /**
   * @brief Construct block-based KV cache manager
   *
   * @param num_blocks Total number of physical blocks available
   * @param block_size Number of tokens per block (typically 16)
   * @param num_layers Number of transformer layers
   * @param num_kv_heads Number of KV heads (for GQA)
   * @param head_size Size of each attention head
   * @param device Device to allocate cache on
   */
  KVCacheManager(i32 num_blocks, i32 block_size, i32 num_layers,
                   i32 num_kv_heads, i32 head_size, DeviceType device);

  /**
   * @brief Initialize cache (allocate GPU memory for all blocks)
   */
  Result<void> init();

  /**
   * @brief Allocate blocks for a new sequence
   *
   * @param seq_id Sequence ID
   * @param num_tokens Number of tokens needed
   * @return Number of blocks allocated
   */
  Result<i32> allocate_sequence(i32 seq_id, i32 num_tokens);

  /**
   * @brief Extend sequence by allocating additional blocks
   *
   * Used during generation when sequence grows beyond current capacity.
   *
   * @param seq_id Sequence ID
   * @param additional_tokens Number of additional tokens needed
   * @return Number of additional blocks allocated
   */
  Result<i32> extend_sequence(i32 seq_id, i32 additional_tokens);

  /**
   * @brief Free all blocks allocated to a sequence
   *
   * @param seq_id Sequence ID
   * @return Number of blocks freed
   */
  Result<i32> free_sequence(i32 seq_id);

  /**
   * @brief Get key cache tensor for a layer
   *
   * @param layer_idx Layer index
   * @return Key cache tensor [num_blocks, num_kv_heads, block_size, head_size]
   */
  Tensor& get_key_cache(i32 layer_idx);

  /**
   * @brief Get value cache tensor for a layer
   *
   * @param layer_idx Layer index
   * @return Value cache tensor [num_blocks, num_kv_heads, block_size, head_size]
   */
  Tensor& get_value_cache(i32 layer_idx);

  /**
   * @brief Get block table in GPU format for a batch of sequences
   *
   * Prepares the block table for GPU kernels:
   * - Flat array: [seq_0_blocks..., seq_1_blocks..., ...]
   * - Padded with -1 to max_blocks_per_seq
   * - Returns CPU tensor that can be copied to GPU
   *
   * @param seq_ids Ordered list of sequence IDs
   * @return CPU Tensor with shape [num_seqs, max_blocks_per_seq], dtype=Int32
   */
  Result<Tensor> get_block_table_tensor(const std::vector<i32>& seq_ids);

  /**
   * @brief Get sequence lengths for a batch
   *
   * @param seq_ids Ordered list of sequence IDs
   * @return Vector of sequence lengths (in tokens)
   */
  Result<std::vector<i32>> get_sequence_lengths(const std::vector<i32>& seq_ids) const;

  /**
   * @brief Get number of blocks allocated to a sequence
   *
   * @param seq_id Sequence ID
   * @return Number of blocks
   */
  Result<i32> get_num_blocks(i32 seq_id) const;

  /**
   * @brief Get current capacity (in tokens) for a sequence
   *
   * @param seq_id Sequence ID
   * @return Capacity in tokens (num_blocks * block_size)
   */
  Result<i32> get_sequence_capacity(i32 seq_id) const;

  /**
   * @brief Check if a sequence is allocated
   */
  bool is_sequence_allocated(i32 seq_id) const;

  /**
   * @brief Update the current token count for a sequence
   *
   * This should be called after writing new tokens to the cache.
   *
   * @param seq_id Sequence ID
   * @param new_token_count New total token count (including newly added tokens)
   * @return Result<void> Success or error
   */
  Result<void> update_sequence_length(i32 seq_id, i32 new_token_count);

  /**
   * @brief Get number of free blocks
   */
  i32 num_free_blocks() const;

  /**
   * @brief Get block manager statistics
   */
  f32 get_block_utilization() const;

  /**
   * @brief Reset all allocations (clear all sequences)
   */
  void reset();

  // Accessors
  [[nodiscard]] i32 num_blocks() const noexcept { return num_blocks_; }
  [[nodiscard]] i32 block_size() const noexcept { return block_size_; }
  [[nodiscard]] i32 num_layers() const noexcept { return num_layers_; }
  [[nodiscard]] i32 num_kv_heads() const noexcept { return num_kv_heads_; }
  [[nodiscard]] i32 head_size() const noexcept { return head_size_; }
  [[nodiscard]] i32 kv_dim() const noexcept { return num_kv_heads_ * head_size_; }

  /**
   * @brief Get maximum blocks per sequence (for GPU kernel configuration)
   *
   * This is computed based on maximum sequence length supported.
   * Used to determine padding size for block table.
   */
  i32 get_max_blocks_per_seq() const;

 private:
  // Configuration
  i32 num_blocks_;      ///< Total number of physical blocks
  i32 block_size_;      ///< Tokens per block
  i32 num_layers_;      ///< Number of transformer layers
  i32 num_kv_heads_;    ///< Number of KV heads
  i32 head_size_;       ///< Size of each head
  DeviceType device_;   ///< Device for allocation

  // Block management
  std::unique_ptr<runtime::BlockManager> block_manager_;
  std::unique_ptr<runtime::BlockTable> block_table_;

  // KV cache tensors: one per layer
  // Shape: [num_blocks, num_kv_heads, block_size, head_size]
  std::vector<Tensor> key_caches_;
  std::vector<Tensor> value_caches_;

  // Track sequence token counts (for sequence length tracking)
  std::unordered_map<i32, i32> seq_num_tokens_;

  bool initialized_ = false;

  /**
   * @brief Calculate number of blocks needed for a given number of tokens
   */
  i32 calculate_num_blocks_needed(i32 num_tokens) const {
    return (num_tokens + block_size_ - 1) / block_size_;
  }
};

}  // namespace photon::model
