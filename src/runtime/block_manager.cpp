/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#include "photon/runtime/block_manager.hpp"
#include <glog/logging.h>
#include <algorithm>

namespace photon::runtime {

BlockManager::BlockManager(i32 num_blocks, i32 block_size, bool thread_safe)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      thread_safe_(thread_safe),
      allocated_(num_blocks, false) {

  // Initialize free blocks stack with all block IDs
  free_blocks_.reserve(num_blocks);
  for (i32 i = 0; i < num_blocks; ++i) {
    free_blocks_.push_back(i);
  }

  VLOG(1) << "BlockManager initialized:";
  VLOG(1) << "  Total blocks: " << num_blocks;
  VLOG(1) << "  Block size: " << block_size << " tokens";
  VLOG(1) << "  Total capacity: " << (num_blocks * block_size) << " tokens";
  VLOG(1) << "  Thread-safe: " << (thread_safe ? "yes" : "no");
}

Result<i32> BlockManager::allocate_block() {
  LockGuard lock(mutex_, thread_safe_);

  if (free_blocks_.empty()) {
    return Err<i32>(ErrorCode::OutOfMemory,
                   "No free blocks available. Total blocks: " +
                   std::to_string(num_blocks_) +
                   ", All allocated.");
  }

  // Pop from free blocks stack
  i32 block_id = free_blocks_.back();
  free_blocks_.pop_back();

  // Mark as allocated
  allocated_[block_id] = true;

  VLOG(2) << "Allocated block " << block_id
          << " (free blocks remaining: " << free_blocks_.size() << ")";

  return Ok(block_id);
}

Result<std::vector<i32>> BlockManager::allocate_blocks(i32 num_blocks_needed) {
  LockGuard lock(mutex_, thread_safe_);

  if (num_blocks_needed <= 0) {
    return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                "Number of blocks must be positive");
  }

  if (static_cast<i32>(free_blocks_.size()) < num_blocks_needed) {
    return Err<std::vector<i32>>(
        ErrorCode::OutOfMemory,
        "Insufficient free blocks. Requested: " +
        std::to_string(num_blocks_needed) +
        ", Available: " + std::to_string(free_blocks_.size()));
  }

  std::vector<i32> allocated_block_ids;
  allocated_block_ids.reserve(num_blocks_needed);

  // Allocate requested number of blocks
  for (i32 i = 0; i < num_blocks_needed; ++i) {
    i32 block_id = free_blocks_.back();
    free_blocks_.pop_back();
    allocated_[block_id] = true;
    allocated_block_ids.push_back(block_id);
  }

  VLOG(2) << "Allocated " << num_blocks_needed << " blocks"
          << " (free blocks remaining: " << free_blocks_.size() << ")";

  return Ok(std::move(allocated_block_ids));
}

Result<void> BlockManager::free_block(i32 block_id) {
  LockGuard lock(mutex_, thread_safe_);

  if (!is_valid_block_id(block_id)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid block ID: " + std::to_string(block_id));
  }

  if (!allocated_[block_id]) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Block " + std::to_string(block_id) +
                    " is not allocated (double free?)");
  }

  // Mark as free
  allocated_[block_id] = false;

  // Push back to free blocks stack
  free_blocks_.push_back(block_id);

  VLOG(2) << "Freed block " << block_id
          << " (free blocks: " << free_blocks_.size() << ")";

  return Ok();
}

Result<void> BlockManager::free_blocks(const std::vector<i32>& block_ids) {
  LockGuard lock(mutex_, thread_safe_);

  // Validate all block IDs first
  for (i32 block_id : block_ids) {
    if (!is_valid_block_id(block_id)) {
      return Err<void>(ErrorCode::InvalidArgument,
                      "Invalid block ID in batch: " + std::to_string(block_id));
    }
    if (!allocated_[block_id]) {
      return Err<void>(ErrorCode::InvalidArgument,
                      "Block " + std::to_string(block_id) +
                      " is not allocated");
    }
  }

  // Free all blocks
  for (i32 block_id : block_ids) {
    allocated_[block_id] = false;
    free_blocks_.push_back(block_id);
  }

  VLOG(2) << "Freed " << block_ids.size() << " blocks"
          << " (free blocks: " << free_blocks_.size() << ")";

  return Ok();
}

i32 BlockManager::get_num_free_blocks() const {
  LockGuard lock(mutex_, thread_safe_);
  return static_cast<i32>(free_blocks_.size());
}

i32 BlockManager::get_num_allocated_blocks() const {
  LockGuard lock(mutex_, thread_safe_);
  return num_blocks_ - static_cast<i32>(free_blocks_.size());
}

bool BlockManager::is_allocated(i32 block_id) const {
  LockGuard lock(mutex_, thread_safe_);
  if (!is_valid_block_id(block_id)) {
    return false;
  }
  return allocated_[block_id];
}

void BlockManager::reset() {
  LockGuard lock(mutex_, thread_safe_);

  // Clear free blocks list
  free_blocks_.clear();
  free_blocks_.reserve(num_blocks_);

  // Reset all blocks to free state
  for (i32 i = 0; i < num_blocks_; ++i) {
    allocated_[i] = false;
    free_blocks_.push_back(i);
  }

  LOG(INFO) << "BlockManager reset: all " << num_blocks_ << " blocks freed";
}

f32 BlockManager::get_utilization() const {
  LockGuard lock(mutex_, thread_safe_);
  if (num_blocks_ == 0) return 0.0f;
  i32 num_allocated = num_blocks_ - static_cast<i32>(free_blocks_.size());
  return static_cast<f32>(num_allocated) / static_cast<f32>(num_blocks_);
}

} // namespace photon::runtime
