/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#include "photon/runtime/block_table.hpp"
#include <glog/logging.h>
#include <algorithm>

namespace photon::runtime {

BlockTable::BlockTable(bool thread_safe)
    : thread_safe_(thread_safe) {
  VLOG(1) << "BlockTable initialized (thread-safe: "
          << (thread_safe ? "yes" : "no") << ")";
}

Result<void> BlockTable::allocate_sequence(i32 seq_id,
                                           const std::vector<i32>& block_ids) {
  LockGuard lock(mutex_, thread_safe_);

  if (seq_to_blocks_.find(seq_id) != seq_to_blocks_.end()) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Sequence " + std::to_string(seq_id) + " already exists");
  }

  seq_to_blocks_[seq_id] = block_ids;

  VLOG(2) << "Allocated sequence " << seq_id << " with " << block_ids.size()
          << " blocks: [";
  if (VLOG_IS_ON(2)) {
    for (size_t i = 0; i < block_ids.size(); ++i) {
      if (i > 0) VLOG(2) << ", ";
      VLOG(2) << block_ids[i];
    }
    VLOG(2) << "]";
  }

  return Ok();
}

Result<void> BlockTable::append_block(i32 seq_id, i32 block_id) {
  LockGuard lock(mutex_, thread_safe_);

  auto it = seq_to_blocks_.find(seq_id);
  if (it == seq_to_blocks_.end()) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Sequence " + std::to_string(seq_id) + " does not exist");
  }

  it->second.push_back(block_id);

  VLOG(2) << "Appended block " << block_id << " to sequence " << seq_id
          << " (now " << it->second.size() << " blocks)";

  return Ok();
}

Result<void> BlockTable::append_blocks(i32 seq_id,
                                       const std::vector<i32>& block_ids) {
  LockGuard lock(mutex_, thread_safe_);

  auto it = seq_to_blocks_.find(seq_id);
  if (it == seq_to_blocks_.end()) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Sequence " + std::to_string(seq_id) + " does not exist");
  }

  it->second.insert(it->second.end(), block_ids.begin(), block_ids.end());

  VLOG(2) << "Appended " << block_ids.size() << " blocks to sequence "
          << seq_id << " (now " << it->second.size() << " blocks)";

  return Ok();
}

Result<std::vector<i32>> BlockTable::get_blocks(i32 seq_id) const {
  LockGuard lock(mutex_, thread_safe_);

  auto it = seq_to_blocks_.find(seq_id);
  if (it == seq_to_blocks_.end()) {
    return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                "Sequence " + std::to_string(seq_id) +
                                " does not exist");
  }

  return Ok(it->second);
}

Result<i32> BlockTable::get_num_blocks(i32 seq_id) const {
  LockGuard lock(mutex_, thread_safe_);

  auto it = seq_to_blocks_.find(seq_id);
  if (it == seq_to_blocks_.end()) {
    return Err<i32>(ErrorCode::InvalidArgument,
                   "Sequence " + std::to_string(seq_id) + " does not exist");
  }

  return Ok(static_cast<i32>(it->second.size()));
}

Result<std::vector<i32>> BlockTable::free_sequence(i32 seq_id) {
  LockGuard lock(mutex_, thread_safe_);

  auto it = seq_to_blocks_.find(seq_id);
  if (it == seq_to_blocks_.end()) {
    return Err<std::vector<i32>>(ErrorCode::InvalidArgument,
                                "Sequence " + std::to_string(seq_id) +
                                " does not exist");
  }

  // Get blocks to be freed
  std::vector<i32> freed_blocks = std::move(it->second);

  // Remove sequence from map
  seq_to_blocks_.erase(it);

  VLOG(2) << "Freed sequence " << seq_id << " (" << freed_blocks.size()
          << " blocks returned)";

  return Ok(std::move(freed_blocks));
}

bool BlockTable::has_sequence(i32 seq_id) const {
  LockGuard lock(mutex_, thread_safe_);
  return seq_to_blocks_.find(seq_id) != seq_to_blocks_.end();
}

std::vector<i32> BlockTable::get_sequence_ids() const {
  LockGuard lock(mutex_, thread_safe_);

  std::vector<i32> seq_ids;
  seq_ids.reserve(seq_to_blocks_.size());

  for (const auto& [seq_id, _] : seq_to_blocks_) {
    seq_ids.push_back(seq_id);
  }

  return seq_ids;
}

i32 BlockTable::get_num_sequences() const {
  LockGuard lock(mutex_, thread_safe_);
  return static_cast<i32>(seq_to_blocks_.size());
}

Result<Tensor> BlockTable::to_gpu_format(const std::vector<i32>& seq_ids,
                                         i32 max_blocks_per_seq) const {
  LockGuard lock(mutex_, thread_safe_);

  if (seq_ids.empty()) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "seq_ids cannot be empty");
  }

  if (max_blocks_per_seq <= 0) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "max_blocks_per_seq must be positive");
  }

  i32 num_seqs = static_cast<i32>(seq_ids.size());

  // Create CPU tensor: [num_seqs, max_blocks_per_seq]
  auto tensor_result = Tensor::create({num_seqs, max_blocks_per_seq},
                                     DataType::Int32,
                                     DeviceType::CPU);
  if (!tensor_result) {
    return Err<Tensor>(tensor_result.error());
  }

  Tensor block_table_cpu = std::move(tensor_result.value());
  i32* data = block_table_cpu.ptr<i32>();

  // Fill with -1 (padding value)
  std::fill(data, data + num_seqs * max_blocks_per_seq, -1);

  // Copy block IDs for each sequence
  for (i32 i = 0; i < num_seqs; ++i) {
    i32 seq_id = seq_ids[i];

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
      return Err<Tensor>(ErrorCode::InvalidArgument,
                        "Sequence " + std::to_string(seq_id) +
                        " not found in block table");
    }

    const auto& blocks = it->second;
    i32 num_blocks = static_cast<i32>(blocks.size());

    if (num_blocks > max_blocks_per_seq) {
      return Err<Tensor>(ErrorCode::InvalidArgument,
                        "Sequence " + std::to_string(seq_id) + " has " +
                        std::to_string(num_blocks) + " blocks, exceeds max " +
                        std::to_string(max_blocks_per_seq));
    }

    // Copy block IDs to flat array
    i32* seq_data = data + i * max_blocks_per_seq;
    for (i32 j = 0; j < num_blocks; ++j) {
      seq_data[j] = blocks[j];
    }
  }

  VLOG(2) << "Converted block table to GPU format: " << num_seqs << " seqs, "
          << max_blocks_per_seq << " max blocks per seq";

  return Ok(std::move(block_table_cpu));
}

void BlockTable::reset() {
  LockGuard lock(mutex_, thread_safe_);

  i32 num_seqs = static_cast<i32>(seq_to_blocks_.size());
  seq_to_blocks_.clear();

  LOG(INFO) << "BlockTable reset: " << num_seqs << " sequences cleared";
}

} // namespace photon::runtime
