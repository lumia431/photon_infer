#ifndef PHOTON_MODEL_CONFIG_HPP
#define PHOTON_MODEL_CONFIG_HPP

#include <cstdint>

namespace photon::model {

// Model configuration read from binary file (POD type for direct fread)
struct ModelConfig {
  int32_t dim = 0;         // Model dimension
  int32_t hidden_dim = 0;  // FFN hidden dimension
  int32_t layer_num = 0;   // Number of layers
  int32_t head_num = 0;    // Number of attention heads
  int32_t kv_head_num = 0; // Number of KV heads (for GQA)
  int32_t vocab_size = 0;  // Vocabulary size (negative = shared weights)
  int32_t seq_len = 0;     // Maximum sequence length
};

// Runtime transformer configuration with derived values
struct TransformerConfig {
  // Derived dimensions
  int32_t kv_dim = 0;      // KV dimension
  int32_t kv_mul = 0;      // Head replication factor for GQA
  int32_t head_size = 0;   // Size per attention head
  int32_t vocab_size = 0;  // Actual vocabulary size

  // Base configuration
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t seq_len = 0;

  bool is_shared_weight = false; // Whether embeddings and output weights are shared

  // Compute derived values from ModelConfig
  static TransformerConfig from_model_config(const ModelConfig& config, int32_t tokenizer_vocab_size);
};

} // namespace photon::model

#endif // PHOTON_MODEL_CONFIG_HPP
