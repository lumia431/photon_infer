/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once


#include <cstdint>
#include "photon/core/types.hpp"

namespace photon::model {

// Model configuration read from binary file (POD type for direct fread)
struct ModelConfig {
  i32 dim = 0;         // Model dimension
  i32 hidden_dim = 0;  // FFN hidden dimension
  i32 layer_num = 0;   // Number of layers
  i32 head_num = 0;    // Number of attention heads
  i32 kv_head_num = 0; // Number of KV heads (for GQA)
  i32 vocab_size = 0;  // Vocabulary size (negative = shared weights)
  i32 seq_len = 0;     // Maximum sequence length
};

// Runtime transformer configuration with derived values
struct TransformerConfig {
  // Base configuration
  i32 dim = 0;
  i32 hidden_dim = 0;
  i32 n_layers = 0;      // Number of layers (renamed from layer_num)
  i32 n_heads = 0;       // Number of attention heads (renamed from head_num)
  i32 n_kv_heads = 0;    // Number of KV heads for GQA (renamed from kv_head_num)
  i32 seq_len = 0;       // Maximum sequence length
  i32 vocab_size = 0;    // Actual vocabulary size
  i32 head_size = 0;     // Size per attention head
  f32 norm_eps = 1e-5f;  // RMSNorm epsilon

  // Derived dimensions
  i32 kv_dim = 0;        // KV dimension (n_kv_heads * head_size)
  i32 kv_mul = 0;        // Head replication factor for GQA (n_heads / n_kv_heads)

  bool is_shared_weight = false;  // Whether embeddings and output weights are shared
  photon::DeviceType device = photon::DeviceType::CPU;  // Device for computation

  // Compute derived values
  void compute_derived() {
    kv_dim = n_kv_heads * head_size;
    kv_mul = n_heads / n_kv_heads;
  }

  // Compute derived values from ModelConfig
  static TransformerConfig from_model_config(const ModelConfig& config, i32 tokenizer_vocab_size);
};

} // namespace photon::model

