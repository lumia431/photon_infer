/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#pragma once

/**
 * @file block_table.hpp
 * @brief Block Table for mapping sequences to physical blocks
 * @version 1.0.0
 *
 * Maintains the virtual-to-physical block mapping for each sequence.
 * Each sequence has a list of physical block IDs that may be non-contiguous.
 */

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"
#include "photon/core/tensor.hpp"
#include <vector>
#include <unordered_map>
#include <mutex>

namespace photon::runtime {

/**
 * @brief Manages virtual-to-physical block mapping for sequences
 *
 * The BlockTable stores the mapping from sequence IDs to their physical blocks.
 * Each sequence can have multiple blocks, and blocks don't need to be contiguous.
 *
 * Example:
 *   Sequence 0: [block 5, block 12, block 3]   (3 blocks, non-contiguous)
 *   Sequence 1: [block 0, block 7]             (2 blocks)
 *   Sequence 2: [block 9, block 2, block 11, block 6]  (4 blocks)
 *
 * GPU Format (flat array with padding):
 *   [5, 12, 3, -1,     // seq 0 (3 blocks, padded to max_blocks_per_seq)
 *    0, 7, -1, -1,     // seq 1 (2 blocks)
 *    9, 2, 11, 6]      // seq 2 (4 blocks)
 */
class BlockTable {
 public:
  /**
   * @brief Construct an empty BlockTable
   *
   * @param thread_safe Enable thread-safe operations (default: false)
   */
  explicit BlockTable(bool thread_safe = false);

  /**
   * @brief Allocate a new sequence with initial blocks
   *
   * @param seq_id Sequence ID
   * @param block_ids Initial physical block IDs for this sequence
   * @return Ok on success, error if seq_id already exists
   */
  Result<void> allocate_sequence(i32 seq_id, const std::vector<i32>& block_ids);

  /**
   * @brief Append a physical block to an existing sequence
   *
   * This is used for dynamic sequence extension during generation.
   *
   * @param seq_id Sequence ID
   * @param block_id Physical block ID to append
   * @return Ok on success, error if sequence doesn't exist
   */
  Result<void> append_block(i32 seq_id, i32 block_id);

  /**
   * @brief Append multiple blocks to a sequence
   *
   * @param seq_id Sequence ID
   * @param block_ids Vector of physical block IDs to append
   * @return Ok on success, error if sequence doesn't exist
   */
  Result<void> append_blocks(i32 seq_id, const std::vector<i32>& block_ids);

  /**
   * @brief Get all physical blocks for a sequence
   *
   * @param seq_id Sequence ID
   * @return Vector of physical block IDs, or error if sequence doesn't exist
   */
  Result<std::vector<i32>> get_blocks(i32 seq_id) const;

  /**
   * @brief Get number of blocks allocated to a sequence
   *
   * @param seq_id Sequence ID
   * @return Number of blocks, or error if sequence doesn't exist
   */
  Result<i32> get_num_blocks(i32 seq_id) const;

  /**
   * @brief Free all blocks from a sequence and remove it
   *
   * @param seq_id Sequence ID
   * @return Vector of freed physical block IDs, or error if sequence doesn't exist
   */
  Result<std::vector<i32>> free_sequence(i32 seq_id);

  /**
   * @brief Check if a sequence exists
   *
   * @param seq_id Sequence ID to check
   * @return true if sequence exists, false otherwise
   */
  bool has_sequence(i32 seq_id) const;

  /**
   * @brief Get all active sequence IDs
   *
   * @return Vector of sequence IDs
   */
  std::vector<i32> get_sequence_ids() const;

  /**
   * @brief Get total number of active sequences
   */
  i32 get_num_sequences() const;

  /**
   * @brief Prepare block table in GPU format (flat array with padding)
   *
   * This converts the block table to a format suitable for GPU kernels:
   * - Flat array: [seq_0_blocks..., seq_1_blocks..., ...]
   * - Each sequence's blocks are padded to max_blocks_per_seq with -1
   * - Shape: [num_seqs, max_blocks_per_seq]
   *
   * @param seq_ids Ordered list of sequence IDs to include
   * @param max_blocks_per_seq Maximum blocks per sequence (for padding)
   * @return Tensor on CPU with shape [num_seqs, max_blocks_per_seq]
   */
  Result<Tensor> to_gpu_format(const std::vector<i32>& seq_ids,
                               i32 max_blocks_per_seq) const;

  /**
   * @brief Clear all sequences
   *
   * WARNING: This does not free the underlying physical blocks.
   * Caller must ensure blocks are properly freed via BlockManager.
   */
  void reset();

 private:
  bool thread_safe_;  ///< Enable thread-safe operations

  /// Mapping: seq_id -> [physical_block_0, physical_block_1, ...]
  std::unordered_map<i32, std::vector<i32>> seq_to_blocks_;

  mutable std::mutex mutex_;  ///< Mutex for thread-safe operations

  /// Lock guard helper for thread-safe operations
  class LockGuard {
   public:
    LockGuard(std::mutex& mutex, bool enabled)
        : mutex_(mutex), enabled_(enabled) {
      if (enabled_) mutex_.lock();
    }
    ~LockGuard() {
      if (enabled_) mutex_.unlock();
    }
   private:
    std::mutex& mutex_;
    bool enabled_;
  };
};

} // namespace photon::runtime
