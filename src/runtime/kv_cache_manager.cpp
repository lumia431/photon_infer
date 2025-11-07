/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file kv_cache_manager.cpp
 * @brief Implementation of KV Cache Manager
 */

#include "photon/runtime/kv_cache_manager.hpp"
#include <glog/logging.h>

namespace photon::model {

KVCacheManager::KVCacheManager(i32 max_sequences, i32 max_seq_len,
                                i32 num_layers, i32 kv_dim, DeviceType device)
    : max_sequences_(max_sequences),
      max_seq_len_(max_seq_len),
      num_layers_(num_layers),
      kv_dim_(kv_dim),
      device_(device),
      slot_free_(max_sequences, true) {

  key_caches_.reserve(num_layers);
  value_caches_.reserve(num_layers);
}

Result<void> KVCacheManager::init() {
  if (initialized_) {
    return Ok();
  }

  LOG(INFO) << "Initializing KV Cache Manager:";
  LOG(INFO) << "  Max sequences: " << max_sequences_;
  LOG(INFO) << "  Max seq len: " << max_seq_len_;
  LOG(INFO) << "  Num layers: " << num_layers_;
  LOG(INFO) << "  KV dim: " << kv_dim_;

  // Calculate total cache size
  i64 total_tokens = static_cast<i64>(max_sequences_) * max_seq_len_;
  i64 total_elements_per_layer = total_tokens * kv_dim_;
  i64 total_bytes_per_layer = total_elements_per_layer * sizeof(f32);
  i64 total_bytes = total_bytes_per_layer * num_layers_ * 2;  // 2 for K and V

  LOG(INFO) << "  Total cache size: " << (total_bytes / (1024.0 * 1024.0))
            << " MB";

  // Allocate cache for each layer
  for (i32 layer = 0; layer < num_layers_; ++layer) {
    // Allocate key cache
    auto key_result = Tensor::create({total_tokens, kv_dim_},
                                     DataType::Float32, device_);
    if (!key_result) {
      LOG(ERROR) << "Failed to allocate key cache for layer " << layer;
      return Err<void>(key_result.error());
    }
    key_caches_.push_back(std::move(key_result.value()));

    // Allocate value cache
    auto value_result = Tensor::create({total_tokens, kv_dim_},
                                       DataType::Float32, device_);
    if (!value_result) {
      LOG(ERROR) << "Failed to allocate value cache for layer " << layer;
      return Err<void>(value_result.error());
    }
    value_caches_.push_back(std::move(value_result.value()));

    // Zero-initialize caches (optional but good practice)
    // key_caches_.back().fill(0.0f);
    // value_caches_.back().fill(0.0f);
  }

  initialized_ = true;
  LOG(INFO) << "KV Cache Manager initialized successfully";

  return Ok();
}

Result<void> KVCacheManager::allocate_sequence(i32 seq_id, i32 num_tokens) {
  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "KV Cache Manager not initialized");
  }

  // Check if sequence already allocated
  if (sequence_map_.find(seq_id) != sequence_map_.end()) {
    auto& info = sequence_map_[seq_id];
    if (info.active) {
      // Already allocated, just verify capacity
      if (num_tokens > max_seq_len_) {
        return Err<void>(ErrorCode::InvalidArgument,
                        "Requested tokens exceed max_seq_len");
      }
      info.num_tokens = std::max(info.num_tokens, num_tokens);
      return Ok();
    }
  }

  // Find a free slot
  i32 slot_idx = -1;
  for (i32 i = 0; i < max_sequences_; ++i) {
    if (slot_free_[i]) {
      slot_idx = i;
      break;
    }
  }

  if (slot_idx == -1) {
    return Err<void>(ErrorCode::OutOfMemory,
                    "No free cache slots available");
  }

  if (num_tokens > max_seq_len_) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Requested tokens exceed max_seq_len");
  }

  // Allocate the slot
  slot_free_[slot_idx] = false;
  sequence_map_[seq_id] = SequenceInfo{slot_idx, num_tokens, true};

  VLOG(1) << "Allocated sequence " << seq_id << " to slot " << slot_idx
          << " (" << num_tokens << " tokens)";

  return Ok();
}

Result<void> KVCacheManager::free_sequence(i32 seq_id) {
  auto it = sequence_map_.find(seq_id);
  if (it == sequence_map_.end() || !it->second.active) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Sequence not found or already freed");
  }

  i32 slot_idx = it->second.slot_idx;
  slot_free_[slot_idx] = true;
  it->second.active = false;

  VLOG(1) << "Freed sequence " << seq_id << " from slot " << slot_idx;

  return Ok();
}

Tensor& KVCacheManager::get_key_cache(i32 layer_idx) {
  CHECK(layer_idx >= 0 && layer_idx < num_layers_)
      << "Invalid layer index: " << layer_idx;
  return key_caches_[layer_idx];
}

Tensor& KVCacheManager::get_value_cache(i32 layer_idx) {
  CHECK(layer_idx >= 0 && layer_idx < num_layers_)
      << "Invalid layer index: " << layer_idx;
  return value_caches_[layer_idx];
}

Result<i32> KVCacheManager::get_sequence_offset(i32 seq_id) const {
  auto it = sequence_map_.find(seq_id);
  if (it == sequence_map_.end() || !it->second.active) {
    return Err<i32>(ErrorCode::InvalidArgument,
                   "Sequence not found or not active");
  }

  i32 slot_idx = it->second.slot_idx;
  i32 offset = slot_idx * max_seq_len_;
  return Ok(offset);
}

bool KVCacheManager::is_sequence_allocated(i32 seq_id) const {
  auto it = sequence_map_.find(seq_id);
  return it != sequence_map_.end() && it->second.active;
}

i32 KVCacheManager::num_free_slots() const {
  i32 count = 0;
  for (bool free : slot_free_) {
    if (free) ++count;
  }
  return count;
}

void KVCacheManager::reset() {
  for (auto& [seq_id, info] : sequence_map_) {
    if (info.active) {
      slot_free_[info.slot_idx] = true;
      info.active = false;
    }
  }
  sequence_map_.clear();
  LOG(INFO) << "KV Cache Manager reset";
}

}  // namespace photon::model
