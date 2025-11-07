/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file checkpoint.hpp
 * @brief Checkpoint file format and loader for LLaMA models
 * @version 0.1.0
 */


#include <memory>
#include <string>
#include "photon/core/error.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"
#include "photon/arch/transformer_block.hpp"

namespace photon::model {

/**
 * @brief Checkpoint file header matching standard format
 *
 * File structure:
 * [Header: 7 × i32] [Weights: float32 array]
 */
struct CheckpointHeader {
  i32 dim;           // Model dimension
  i32 hidden_dim;    // FFN hidden dimension
  i32 n_layers;      // Number of transformer layers
  i32 n_heads;       // Number of attention heads
  i32 n_kv_heads;    // Number of KV heads (for GQA)
  i32 vocab_size;    // Vocabulary size
  i32 seq_len;       // Maximum sequence length
};

/**
 * @brief Checkpoint loader using memory-mapped I/O
 *
 * Loads weights from binary checkpoint file in standard format:
 * - Header: 7 int32 values (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
 * - Weights: float32 array in specific order
 *
 * Weight order:
 * 1. Token embedding: [vocab_size × dim]
 * 2. For each layer:
 *    - attn_norm: [dim]
 *    - wq: [dim × dim]
 *    - wk: [kv_dim × dim]
 *    - wv: [kv_dim × dim]
 *    - wo: [dim × dim]
 *    - ffn_norm: [dim]
 *    - w1: [hidden_dim × dim]
 *    - w2: [dim × hidden_dim]
 *    - w3: [hidden_dim × dim]
 * 3. Final RMSNorm: [dim]
 * 4. Classifier (optional, if not weight sharing): [vocab_size × dim]
 */
class CheckpointLoader {
 public:
  /**
   * @brief Open checkpoint file
   *
   * @param checkpoint_path Path to checkpoint file
   * @return Result<void> Success or error
   */
  static Result<std::unique_ptr<CheckpointLoader>> open(const std::string& checkpoint_path);

  /**
   * @brief Destructor - unmaps file
   */
  ~CheckpointLoader();

  // Delete copy/move
  CheckpointLoader(const CheckpointLoader&) = delete;
  CheckpointLoader& operator=(const CheckpointLoader&) = delete;
  CheckpointLoader(CheckpointLoader&&) = delete;
  CheckpointLoader& operator=(CheckpointLoader&&) = delete;

  /**
   * @brief Get checkpoint header
   */
  [[nodiscard]] const CheckpointHeader& header() const noexcept {
    return header_;
  }

  /**
   * @brief Load weights into model
   *
   * @param model Target LLaMA model
   * @return Result<void> Success or error
   */
  Result<void> load_weights(class LLaMAModel& model) const;

 private:
  explicit CheckpointLoader() = default;

  /**
   * @brief Read header from file
   */
  Result<void> read_header();

  /**
   * @brief Get pointer to weight data at offset
   *
   * @param offset Offset in float32 elements
   * @return Pointer to weight data
   */
  [[nodiscard]] const f32* weight_ptr(usize offset) const noexcept {
    return weight_data_ + offset;
  }

  CheckpointHeader header_;
  i32 fd_ = -1;           // File descriptor
  void* mmap_data_ = nullptr;  // Memory-mapped file
  usize file_size_ = 0;        // Total file size in bytes
  const f32* weight_data_ = nullptr;  // Pointer to weight array (after header)
};

}  // namespace photon::model

