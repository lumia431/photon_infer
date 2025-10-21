/**
 * @file llama_infer.cpp
 * @brief LLaMA inference demo - interactive text generation
 * @version 0.1.0
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include "photon/core/tensor.hpp"
#include "photon/model/checkpoint.hpp"
#include "photon/model/llama_model.hpp"
#include "photon/model/tokenizer.hpp"

using namespace photon;
using namespace photon::model;

/**
 * @brief Generate text from prompt
 *
 * @param model LLaMA model
 * @param tokenizer Tokenizer
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param print_output Whether to print generated tokens
 * @return Number of tokens generated
 */
i32 generate(LLaMAModel& model, const TikTokenizer& tokenizer,
             const std::string& prompt, i32 max_tokens, bool print_output = true) {
  // Encode prompt
  auto tokens = tokenizer.encode(prompt);
  if (tokens.empty()) {
    std::cerr << "Error: Failed to encode prompt\n";
    return 0;
  }

  // Add BOS token at the beginning (following KuiperInfer behavior)
  tokens.insert(tokens.begin(), tokenizer.bos_id());

  i32 prompt_len = static_cast<i32>(tokens.size());
  if (print_output) {
    std::cout << "Prompt tokens: " << prompt_len << "\n";
    std::cout << "Token IDs: ";
    for (auto token : tokens) {
      std::cout << token << " ";
    }
    std::cout << "\n";
    std::cout << "Generating: " << std::flush;
  }

  // Create logits buffer
  const auto& config = model.config();
  auto logits_result = Tensor::create({config.vocab_size}, DataType::Float32, DeviceType::CPU);
  if (!logits_result) {
    std::cerr << "Error: Failed to create logits buffer\n";
    return 0;
  }
  Tensor logits = std::move(logits_result.value());

  i32 pos = 0;
  i32 next_token = -1;

  // Process tokens (prompt + generation)
  while (pos < max_tokens) {
    i32 current_token;

    if (pos < prompt_len) {
      // Prompt phase: use provided tokens
      current_token = tokens[pos];
    } else {
      // Generation phase: use predicted token
      current_token = next_token;

      // Decode and print if in generation phase
      if (print_output) {
        auto token_text = tokenizer.decode_token(current_token);
        std::cout << token_text << std::flush;
      }
    }

    // Forward pass
    auto forward_result = model.forward(current_token, pos, logits);
    if (!forward_result) {
      std::cerr << "\nError during forward pass: " << forward_result.error().message() << "\n";
      return pos;
    }

    // Sample next token with temperature sampling (T=0.8)
    const f32 temperature = 0.8f;
    const f32* logits_ptr = logits.ptr<f32>();

    // Apply temperature
    std::vector<f32> probs(config.vocab_size);
    f32 max_logit = *std::max_element(logits_ptr, logits_ptr + config.vocab_size);

    f32 sum_exp = 0.0f;
    for (i32 i = 0; i < config.vocab_size; ++i) {
      probs[i] = std::exp((logits_ptr[i] - max_logit) / temperature);
      sum_exp += probs[i];
    }

    // Normalize to get probabilities
    for (i32 i = 0; i < config.vocab_size; ++i) {
      probs[i] /= sum_exp;
    }

    // Sample from distribution
    f32 rand_val = static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX);
    f32 cumulative = 0.0f;
    for (i32 i = 0; i < config.vocab_size; ++i) {
      cumulative += probs[i];
      if (rand_val <= cumulative) {
        next_token = i;
        break;
      }
    }

    // Check for EOS token
    if (next_token == tokenizer.eos_id() && pos >= prompt_len) {
      if (print_output) {
        std::cout << std::flush;
      }
      break;
    }

    pos++;
  }

  if (print_output) {
    std::cout << "\n";
  }

  return pos;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <checkpoint_path> <tokenizer_path>\n";
    std::cerr << "Example: " << argv[0] << " model.bin tokenizer.model\n";
    return 1;
  }

  // Initialize random seed
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const std::string checkpoint_path = argv[1];
  const std::string tokenizer_path = argv[2];

  std::cout << "PhotonInfer - LLaMA Inference Demo\n";
  std::cout << "===================================\n\n";

  // Load tokenizer
  std::cout << "Loading tokenizer from: " << tokenizer_path << "\n";
  auto tokenizer_result = TikTokenizer::load(tokenizer_path);
  if (!tokenizer_result) {
    std::cerr << "Error loading tokenizer: " << tokenizer_result.error().message() << "\n";
    return 1;
  }
  TikTokenizer tokenizer = std::move(tokenizer_result.value());
  std::cout << "  Vocabulary size: " << tokenizer.vocab_size() << "\n";
  std::cout << "  BOS token: " << tokenizer.bos_id() << "\n";
  std::cout << "  EOS token: " << tokenizer.eos_id() << "\n\n";

  // Load checkpoint
  std::cout << "Loading checkpoint from: " << checkpoint_path << "\n";
  auto loader_result = CheckpointLoader::open(checkpoint_path);
  if (!loader_result) {
    std::cerr << "Error loading checkpoint: " << loader_result.error().message() << "\n";
    return 1;
  }
  auto loader = std::move(loader_result.value());

  const auto& header = loader->header();
  std::cout << "Model configuration:\n";
  std::cout << "  Dimension: " << header.dim << "\n";
  std::cout << "  Hidden dim: " << header.hidden_dim << "\n";
  std::cout << "  Layers: " << header.n_layers << "\n";
  std::cout << "  Heads: " << header.n_heads << "\n";
  std::cout << "  KV heads: " << header.n_kv_heads << "\n";
  std::cout << "  Vocab size: " << header.vocab_size << "\n";
  std::cout << "  Sequence length: " << header.seq_len << "\n\n";

  // Create model config
  TransformerConfig config;
  config.dim = header.dim;
  config.hidden_dim = header.hidden_dim;
  config.n_layers = header.n_layers;
  config.n_heads = header.n_heads;
  config.n_kv_heads = header.n_kv_heads;
  config.vocab_size = header.vocab_size;
  config.seq_len = header.seq_len;
  config.head_size = header.dim / header.n_heads;
  config.norm_eps = 1e-5f;
  config.compute_derived();

  // Create model
  std::cout << "Creating model...\n";
  LLaMAModel model(config);

  // Load weights
  std::cout << "Loading weights...\n";
  auto load_result = loader->load_weights(model);
  if (!load_result) {
    std::cerr << "Error loading weights: " << load_result.error().message() << "\n";
    return 1;
  }

  // Initialize model
  std::cout << "Initializing model...\n";
  auto init_result = model.init();
  if (!init_result) {
    std::cerr << "Error initializing model: " << init_result.error().message() << "\n";
    return 1;
  }

  std::cout << "\nModel ready!\n\n";

  // Use fixed prompt for debugging
  const std::string prompt = "What is your name?";
  std::cout << "Using fixed prompt: \"" << prompt << "\"\n\n";

  // Generate response
  auto start = std::chrono::steady_clock::now();
  i32 tokens_generated = generate(model, tokenizer, prompt, 256, true);
  auto end = std::chrono::steady_clock::now();

  auto duration = std::chrono::duration<double>(end - start).count();
  std::cout << "\nGenerated " << tokens_generated << " tokens in "
            << duration << " seconds ("
            << (static_cast<double>(tokens_generated) / duration) << " tokens/s)\n\n";

  return 0;
}
