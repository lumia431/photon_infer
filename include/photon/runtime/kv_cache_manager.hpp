/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file kv_cache_manager.hpp
 * @brief Simple KV Cache Manager for batched inference
 * @version 1.0.0
 *
 * This is a simplified cache manager that allocates contiguous memory blocks
 * for each sequence. Future versions will implement PagedAttention.
 */


#include "photon/core/tensor.hpp"
#include "photon/core/error.hpp"
#include <vector>
#include <unordered_map>

namespace photon::model {

/**
 * @brief Simple KV Cache Manager (contiguous allocation)
 *
 * This manager allocates and tracks KV cache for multiple sequences.
 * Each sequence gets a contiguous chunk of the pre-allocated cache.
 *
 * **Current design (v1):**
 * - Pre-allocates fixed-size cache: [max_sequences, max_seq_len, kv_dim]
 * - Simple slot-based allocation (no paging yet)
 * - Supports concurrent sequences with independent lifetimes
 *
 * **Future (v2 - PagedAttention):**
 * - Block-based allocation with variable-sized blocks
 * - Block tables for non-contiguous memory
 * - Prefix caching and sharing
 */
class KVCacheManager {
 public:
  /**
   * @brief Construct KV cache manager
   *
   * @param max_sequences Maximum number of concurrent sequences
   * @param max_seq_len Maximum sequence length
   * @param num_layers Number of transformer layers
   * @param kv_dim KV dimension (num_kv_heads * head_size)
   * @param device Device to allocate cache on
   */
  KVCacheManager(i32 max_sequences, i32 max_seq_len, i32 num_layers,
                 i32 kv_dim, DeviceType device);

  /**
   * @brief Initialize cache (allocate GPU memory)
   */
  Result<void> init();

  /**
   * @brief Allocate cache for a new sequence
   *
   * @param seq_id Sequence ID (user-provided, typically 0-based index)
   * @param num_tokens Number of tokens to allocate slots for
   * @return Result indicating success or error
   */
  Result<void> allocate_sequence(i32 seq_id, i32 num_tokens);

  /**
   * @brief Free cache for a sequence
   *
   * @param seq_id Sequence ID to free
   * @return Result indicating success or error
   */
  Result<void> free_sequence(i32 seq_id);

  /**
   * @brief Get key cache tensor for a layer
   *
   * @param layer_idx Layer index
   * @return Key cache tensor [max_sequences * max_seq_len, kv_dim]
   */
  Tensor& get_key_cache(i32 layer_idx);

  /**
   * @brief Get value cache tensor for a layer
   *
   * @param layer_idx Layer index
   * @return Value cache tensor [max_sequences * max_seq_len, kv_dim]
   */
  Tensor& get_value_cache(i32 layer_idx);

  /**
   * @brief Get cache offset for a sequence
   *
   * This returns the starting offset in the cache for the given sequence.
   * For contiguous allocation: offset = seq_slot * max_seq_len
   *
   * @param seq_id Sequence ID
   * @return Offset in tokens
   */
  Result<i32> get_sequence_offset(i32 seq_id) const;

  /**
   * @brief Check if a sequence is allocated
   */
  bool is_sequence_allocated(i32 seq_id) const;

  /**
   * @brief Get number of free slots
   */
  i32 num_free_slots() const;

  /**
   * @brief Reset all allocations (clear all sequences)
   */
  void reset();

  // Accessors
  [[nodiscard]] i32 max_sequences() const noexcept { return max_sequences_; }
  [[nodiscard]] i32 max_seq_len() const noexcept { return max_seq_len_; }
  [[nodiscard]] i32 num_layers() const noexcept { return num_layers_; }
  [[nodiscard]] i32 kv_dim() const noexcept { return kv_dim_; }

 private:
  // Configuration
  i32 max_sequences_;
  i32 max_seq_len_;
  i32 num_layers_;
  i32 kv_dim_;
  DeviceType device_;

  // KV cache tensors: one per layer
  // Shape: [max_sequences * max_seq_len, kv_dim]
  std::vector<Tensor> key_caches_;
  std::vector<Tensor> value_caches_;

  // Allocation tracking
  struct SequenceInfo {
    i32 slot_idx;       // Which slot (0 to max_sequences-1)
    i32 num_tokens;     // Number of allocated tokens
    bool active;        // Is this sequence active?
  };

  std::unordered_map<i32, SequenceInfo> sequence_map_;  // seq_id -> info
  std::vector<bool> slot_free_;  // slot availability: true = free

  bool initialized_ = false;
};

}  // namespace photon::model

