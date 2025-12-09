/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#include "photon/runtime/kv_cache_manager.hpp"
#include <glog/logging.h>
#include <algorithm>

namespace photon::model {

KVCacheManager::KVCacheManager(i32 num_blocks, i32 block_size,
                                   i32 num_layers, i32 num_kv_heads,
                                   i32 head_size, DeviceType device)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      device_(device) {

  // Initialize block manager
  block_manager_ = std::make_unique<runtime::BlockManager>(num_blocks, block_size);

  // Initialize block table
  block_table_ = std::make_unique<runtime::BlockTable>();

  LOG(INFO) << "KVCacheManager created:";
  LOG(INFO) << "  Num blocks: " << num_blocks;
  LOG(INFO) << "  Block size: " << block_size << " tokens";
  LOG(INFO) << "  Total capacity: " << (num_blocks * block_size) << " tokens";
  LOG(INFO) << "  Num layers: " << num_layers;
  LOG(INFO) << "  Num KV heads: " << num_kv_heads;
  LOG(INFO) << "  Head size: " << head_size;
  LOG(INFO) << "  KV dim: " << (num_kv_heads * head_size);
}

Result<void> KVCacheManager::init() {
  if (initialized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "KVCacheManager already initialized");
  }

  // Calculate total cache size
  // Shape per layer: [num_blocks, num_kv_heads, block_size, head_size]
  i64 elements_per_block = static_cast<i64>(num_kv_heads_) * block_size_ * head_size_;
  i64 elements_per_layer = elements_per_block * num_blocks_;
  i64 bytes_per_layer = elements_per_layer * sizeof(f32);
  i64 total_bytes = bytes_per_layer * num_layers_ * 2;  // K and V

  LOG(INFO) << "Initializing block-based KV cache:";
  LOG(INFO) << "  Elements per block: " << elements_per_block;
  LOG(INFO) << "  Elements per layer: " << elements_per_layer;
  LOG(INFO) << "  Bytes per layer: " << (bytes_per_layer / (1024.0 * 1024.0)) << " MB";
  LOG(INFO) << "  Total cache size: " << (total_bytes / (1024.0 * 1024.0)) << " MB";

  // Allocate cache for each layer
  key_caches_.reserve(num_layers_);
  value_caches_.reserve(num_layers_);

  for (i32 layer = 0; layer < num_layers_; ++layer) {
    // Allocate key cache: [num_blocks, num_kv_heads, block_size, head_size]
    auto key_result = Tensor::create(
        {num_blocks_, num_kv_heads_, block_size_, head_size_},
        DataType::Float32, device_);
    if (!key_result) {
      return Err<void>(key_result.error());
    }
    key_caches_.push_back(std::move(key_result.value()));

    // Allocate value cache: [num_blocks, num_kv_heads, block_size, head_size]
    auto value_result = Tensor::create(
        {num_blocks_, num_kv_heads_, block_size_, head_size_},
        DataType::Float32, device_);
    if (!value_result) {
      return Err<void>(value_result.error());
    }
    value_caches_.push_back(std::move(value_result.value()));
  }

  initialized_ = true;
  LOG(INFO) << "KV cache initialization complete";
  return Ok();
}

Result<i32> KVCacheManager::allocate_sequence(i32 seq_id, i32 num_tokens) {
  if (!initialized_) {
    return Err<i32>(ErrorCode::InvalidOperator,
                   "KVCacheManager not initialized");
  }

  if (block_table_->has_sequence(seq_id)) {
    return Err<i32>(ErrorCode::InvalidArgument,
                   "Sequence " + std::to_string(seq_id) + " already allocated");
  }

  // Calculate number of blocks needed
  i32 num_blocks_needed = calculate_num_blocks_needed(num_tokens);

  // Allocate blocks from block manager
  auto blocks_result = block_manager_->allocate_blocks(num_blocks_needed);
  if (!blocks_result) {
    return Err<i32>(blocks_result.error());
  }

  std::vector<i32> block_ids = std::move(blocks_result.value());

  // Register blocks in block table
  auto table_result = block_table_->allocate_sequence(seq_id, block_ids);
  if (!table_result) {
    // Rollback: free allocated blocks
    block_manager_->free_blocks(block_ids);
    return Err<i32>(table_result.error());
  }

  // Track sequence token count
  seq_num_tokens_[seq_id] = num_tokens;

  VLOG(1) << "Allocated sequence " << seq_id << ": " << num_tokens
          << " tokens, " << num_blocks_needed << " blocks";

  return Ok(num_blocks_needed);
}

Result<i32> KVCacheManager::extend_sequence(i32 seq_id,
                                              i32 additional_tokens) {
  if (!initialized_) {
    return Err<i32>(ErrorCode::InvalidOperator,
                   "KVCacheManager not initialized");
  }

  if (!block_table_->has_sequence(seq_id)) {
    return Err<i32>(ErrorCode::InvalidArgument,
                   "Sequence " + std::to_string(seq_id) + " not found");
  }

  // Get current allocation
  auto current_blocks_result = block_table_->get_num_blocks(seq_id);
  if (!current_blocks_result) {
    return Err<i32>(current_blocks_result.error());
  }

  i32 current_num_blocks = current_blocks_result.value();
  i32 current_capacity = current_num_blocks * block_size_;

  // Get current token count
  i32 current_tokens = seq_num_tokens_[seq_id];
  i32 new_total_tokens = current_tokens + additional_tokens;

  // Calculate total blocks needed
  i32 total_blocks_needed = calculate_num_blocks_needed(new_total_tokens);
  i32 additional_blocks_needed = total_blocks_needed - current_num_blocks;

  if (additional_blocks_needed <= 0) {
    // Current allocation is sufficient
    seq_num_tokens_[seq_id] = new_total_tokens;
    VLOG(2) << "Sequence " << seq_id << " extended to " << new_total_tokens
            << " tokens (no new blocks needed)";
    return Ok(0);
  }

  // Allocate additional blocks
  auto blocks_result = block_manager_->allocate_blocks(additional_blocks_needed);
  if (!blocks_result) {
    return Err<i32>(blocks_result.error());
  }

  std::vector<i32> new_block_ids = std::move(blocks_result.value());

  // Append blocks to sequence
  auto append_result = block_table_->append_blocks(seq_id, new_block_ids);
  if (!append_result) {
    // Rollback: free newly allocated blocks
    block_manager_->free_blocks(new_block_ids);
    return Err<i32>(append_result.error());
  }

  // Update token count
  seq_num_tokens_[seq_id] = new_total_tokens;

  VLOG(1) << "Extended sequence " << seq_id << " by " << additional_tokens
          << " tokens (" << additional_blocks_needed << " new blocks)";

  return Ok(additional_blocks_needed);
}

Result<i32> KVCacheManager::free_sequence(i32 seq_id) {
  if (!block_table_->has_sequence(seq_id)) {
    return Err<i32>(ErrorCode::InvalidArgument,
                   "Sequence " + std::to_string(seq_id) + " not found");
  }

  // Get blocks from block table
  auto blocks_result = block_table_->free_sequence(seq_id);
  if (!blocks_result) {
    return Err<i32>(blocks_result.error());
  }

  std::vector<i32> block_ids = std::move(blocks_result.value());
  i32 num_blocks_freed = static_cast<i32>(block_ids.size());

  // Free blocks in block manager
  auto free_result = block_manager_->free_blocks(block_ids);
  if (!free_result) {
    LOG(ERROR) << "Failed to free blocks for sequence " << seq_id
               << ": " << free_result.error().message();
    return Err<i32>(free_result.error());
  }

  // Remove from token count tracking
  seq_num_tokens_.erase(seq_id);

  VLOG(1) << "Freed sequence " << seq_id << " (" << num_blocks_freed << " blocks)";

  return Ok(num_blocks_freed);
}

Tensor& KVCacheManager::get_key_cache(i32 layer_idx) {
  return key_caches_[layer_idx];
}

Tensor& KVCacheManager::get_value_cache(i32 layer_idx) {
  return value_caches_[layer_idx];
}

Result<Tensor> KVCacheManager::get_block_table_tensor(
    const std::vector<i32>& seq_ids) {
  if (seq_ids.empty()) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "seq_ids cannot be empty");
  }

  // Calculate max blocks per sequence
  i32 max_blocks_per_seq = get_max_blocks_per_seq();

  // Get block table in GPU format (CPU tensor)
  return block_table_->to_gpu_format(seq_ids, max_blocks_per_seq);
}

Result<std::vector<i32>> KVCacheManager::get_sequence_lengths(
    const std::vector<i32>& seq_ids) const {
  std::vector<i32> lengths;
  lengths.reserve(seq_ids.size());

  for (i32 seq_id : seq_ids) {
    auto it = seq_num_tokens_.find(seq_id);
    if (it == seq_num_tokens_.end()) {
      return Err<std::vector<i32>>(
          ErrorCode::InvalidArgument,
          "Sequence " + std::to_string(seq_id) + " not found");
    }
    lengths.push_back(it->second);
  }

  return Ok(std::move(lengths));
}

Result<void> KVCacheManager::update_sequence_length(i32 seq_id, i32 new_token_count) {
  auto it = seq_num_tokens_.find(seq_id);
  if (it == seq_num_tokens_.end()) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Sequence " + std::to_string(seq_id) + " not allocated");
  }

  // Validate against capacity
  auto capacity_result = get_sequence_capacity(seq_id);
  if (!capacity_result) {
    return Err<void>(capacity_result.error());
  }

  if (new_token_count > capacity_result.value()) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Token count " + std::to_string(new_token_count) +
                    " exceeds capacity " + std::to_string(capacity_result.value()));
  }

  seq_num_tokens_[seq_id] = new_token_count;
  return Ok();
}

Result<i32> KVCacheManager::get_num_blocks(i32 seq_id) const {
  return block_table_->get_num_blocks(seq_id);
}

Result<i32> KVCacheManager::get_sequence_capacity(i32 seq_id) const {
  auto num_blocks_result = block_table_->get_num_blocks(seq_id);
  if (!num_blocks_result) {
    return Err<i32>(num_blocks_result.error());
  }
  return Ok(num_blocks_result.value() * block_size_);
}

bool KVCacheManager::is_sequence_allocated(i32 seq_id) const {
  return block_table_->has_sequence(seq_id);
}

i32 KVCacheManager::num_free_blocks() const {
  return block_manager_->get_num_free_blocks();
}

f32 KVCacheManager::get_block_utilization() const {
  return block_manager_->get_utilization();
}

void KVCacheManager::reset() {
  // Get all active sequences
  auto seq_ids = block_table_->get_sequence_ids();

  // Free all sequences (this will free blocks in block manager)
  for (i32 seq_id : seq_ids) {
    auto result = free_sequence(seq_id);
    if (!result) {
      LOG(WARNING) << "Failed to free sequence " << seq_id
                   << " during reset: " << result.error().message();
    }
  }

  // Clear token count tracking
  seq_num_tokens_.clear();

  // Double-check: reset block manager and table
  block_manager_->reset();
  block_table_->reset();

  LOG(INFO) << "KVCacheManager reset complete";
}

i32 KVCacheManager::get_max_blocks_per_seq() const {
  // Conservative estimate: assume maximum sequence length of 4096 tokens
  // In practice, this should be configurable or computed from actual usage
  const i32 max_seq_len = 4096;
  return (max_seq_len + block_size_ - 1) / block_size_;
}

}  // namespace photon::model
