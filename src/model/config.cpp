#include "photon/model/config.hpp"
#include <cstdlib>

namespace photon::model {

TransformerConfig TransformerConfig::from_model_config(
    const ModelConfig& config,
    int32_t tokenizer_vocab_size) {
  TransformerConfig tf_config;

  // Copy base configuration
  tf_config.dim = config.dim;
  tf_config.hidden_dim = config.hidden_dim;
  tf_config.layer_num = config.layer_num;
  tf_config.head_num = config.head_num;
  tf_config.kv_head_num = config.kv_head_num;
  tf_config.seq_len = config.seq_len;

  // Compute derived dimensions
  tf_config.kv_dim = (config.dim * config.kv_head_num) / config.head_num;
  tf_config.kv_mul = config.head_num / config.kv_head_num;
  tf_config.head_size = config.dim / config.head_num;

  // Handle vocabulary size and shared weights
  // Negative vocab_size indicates shared embeddings/output weights
  if (config.vocab_size > 0) {
    tf_config.is_shared_weight = true;
    tf_config.vocab_size = config.vocab_size;
  } else {
    tf_config.is_shared_weight = false;
    tf_config.vocab_size = std::abs(config.vocab_size);
  }

  // For Llama3 models, allow model vocab_size >= tokenizer vocab_size
  // The difference accounts for special tokens
  if (tf_config.vocab_size < tokenizer_vocab_size) {
    // Use tokenizer size if model size is smaller
    tf_config.vocab_size = tokenizer_vocab_size;
  }

  return tf_config;
}

} // namespace photon::model
