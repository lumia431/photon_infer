/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#pragma once

/**
 * @file block_manager.hpp
 * @brief Block Manager for Paged KV Cache
 * @version 1.0.0
 *
 * Manages physical block allocation and deallocation for paged attention.
 * Inspired by vLLM's BlockAllocator design.
 */

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"
#include <vector>
#include <mutex>

namespace photon::runtime {

/**
 * @brief Manages physical blocks for KV cache
 *
 * The BlockManager maintains a pool of physical blocks and handles
 * allocation/deallocation requests. Each block contains a fixed number
 * of tokens (typically 16).
 *
 * Design:
 * - Free blocks are maintained in a simple vector stack
 * - Allocation: pop from free list
 * - Deallocation: push to free list
 * - Thread-safe operations (optional, controlled by thread_safe flag)
 */
class BlockManager {
 public:
  /**
   * @brief Construct a BlockManager
   *
   * @param num_blocks Total number of physical blocks available
   * @param block_size Number of tokens per block (typically 16)
   * @param thread_safe Enable thread-safe operations (default: false)
   */
  BlockManager(i32 num_blocks, i32 block_size, bool thread_safe = false);

  /**
   * @brief Allocate a single physical block
   *
   * @return Physical block ID (0 to num_blocks-1), or error if no free blocks
   */
  Result<i32> allocate_block();

  /**
   * @brief Allocate multiple consecutive or non-consecutive blocks
   *
   * @param num_blocks_needed Number of blocks to allocate
   * @return Vector of physical block IDs, or error if insufficient blocks
   */
  Result<std::vector<i32>> allocate_blocks(i32 num_blocks_needed);

  /**
   * @brief Free a single physical block
   *
   * @param block_id Physical block ID to free
   * @return Ok on success, error if block_id is invalid or already free
   */
  Result<void> free_block(i32 block_id);

  /**
   * @brief Free multiple physical blocks
   *
   * @param block_ids Vector of physical block IDs to free
   * @return Ok on success, error if any block_id is invalid
   */
  Result<void> free_blocks(const std::vector<i32>& block_ids);

  /**
   * @brief Get number of free blocks available
   */
  i32 get_num_free_blocks() const;

  /**
   * @brief Get total number of blocks
   */
  i32 get_num_total_blocks() const { return num_blocks_; }

  /**
   * @brief Get block size (tokens per block)
   */
  i32 get_block_size() const { return block_size_; }

  /**
   * @brief Get number of allocated blocks
   */
  i32 get_num_allocated_blocks() const;

  /**
   * @brief Check if a block is allocated
   *
   * @param block_id Physical block ID to check
   * @return true if allocated, false if free or invalid
   */
  bool is_allocated(i32 block_id) const;

  /**
   * @brief Reset all blocks to free state
   *
   * WARNING: This should only be called when no sequences are active.
   * Does not check for active references.
   */
  void reset();

  /**
   * @brief Get utilization statistics
   *
   * @return Percentage of blocks allocated (0.0 to 1.0)
   */
  f32 get_utilization() const;

 private:
  i32 num_blocks_;              ///< Total number of blocks
  i32 block_size_;              ///< Tokens per block
  bool thread_safe_;            ///< Enable thread-safe operations

  std::vector<i32> free_blocks_;   ///< Stack of free block IDs
  std::vector<bool> allocated_;    ///< Track allocation status (true=allocated)

  mutable std::mutex mutex_;       ///< Mutex for thread-safe operations

  /**
   * @brief Validate block ID range
   */
  bool is_valid_block_id(i32 block_id) const {
    return block_id >= 0 && block_id < num_blocks_;
  }

  /**
   * @brief Lock guard helper for thread-safe operations
   */
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
