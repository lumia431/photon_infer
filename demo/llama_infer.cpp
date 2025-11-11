/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file llama_infer.cpp
 * @brief LLaMA inference demo - interactive text generation
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "photon/core/tensor.hpp"
#include "photon/io/checkpoint.hpp"
#include "photon/arch/llama_model.hpp"
#include "photon/io/tokenizer.hpp"

using namespace photon;
using namespace photon::model;

struct GenerationState {
  i32 current_pos = 0;
  std::vector<i32> all_tokens;
  Tensor logits;
};

i32 generate_turn(LLaMAModel& model, const TikTokenizer& tokenizer,
                  GenerationState& state, const std::string& prompt,
                  i32 max_new_tokens, bool print_output = true) {
  auto new_tokens = tokenizer.encode(prompt);
  if (new_tokens.empty()) {
    std::cerr << "Error: Failed to encode prompt\n";
    return state.current_pos;
  }

  // Add BOS token only for the first turn
  if (state.all_tokens.empty()) {
    state.all_tokens.push_back(tokenizer.bos_id());
  }

  // Append new prompt tokens
  i32 start_pos = state.current_pos;
  state.all_tokens.insert(state.all_tokens.end(), new_tokens.begin(), new_tokens.end());

  if (print_output) {
    std::cout << "Prompt tokens: " << new_tokens.size() << "\n";
    std::cout << "Total context: " << state.all_tokens.size() << " tokens\n";
    std::cout << "Generating: " << std::flush;
  }

  const auto& config = model.config();
  const f32 temperature = 0.8f;
  i32 tokens_generated = 0;
  i32 next_token = -1;

  // Process prompt tokens
  for (size_t i = start_pos; i < state.all_tokens.size(); ++i) {
    auto forward_result = model.forward(state.all_tokens[i], state.current_pos, state.logits);
    if (!forward_result) {
      std::cerr << "\nError during forward pass: " << forward_result.error().message() << "\n";
      return state.current_pos;
    }
    state.current_pos++;
  }

  // Generate new tokens
  while (tokens_generated < max_new_tokens) {
    const f32* logits_ptr = state.logits.ptr<f32>();
    std::vector<f32> probs(config.vocab_size);
    f32 max_logit = *std::max_element(logits_ptr, logits_ptr + config.vocab_size);

    f32 sum_exp = 0.0f;
    for (i32 i = 0; i < config.vocab_size; ++i) {
      probs[i] = std::exp((logits_ptr[i] - max_logit) / temperature);
      sum_exp += probs[i];
    }

    for (i32 i = 0; i < config.vocab_size; ++i) {
      probs[i] /= sum_exp;
    }

    f32 rand_val = static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX);
    f32 cumulative = 0.0f;
    next_token = -1;
    for (i32 i = 0; i < config.vocab_size; ++i) {
      cumulative += probs[i];
      if (rand_val <= cumulative) {
        next_token = i;
        break;
      }
    }

    if (next_token == tokenizer.eos_id()) {
      if (print_output) std::cout << std::flush;
      break;
    }

    if (print_output) {
      std::cout << tokenizer.decode_token(next_token) << std::flush;
    }

    state.all_tokens.push_back(next_token);
    tokens_generated++;

    auto forward_result = model.forward(next_token, state.current_pos, state.logits);
    if (!forward_result) {
      std::cerr << "\nError during forward pass: " << forward_result.error().message() << "\n";
      return state.current_pos;
    }
    state.current_pos++;
  }

  if (print_output) std::cout << "\n";
  return tokens_generated;
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <checkpoint_path> <tokenizer_path> [device] [options]\n";
    std::cerr << "  device: cpu (default) or cuda\n";
    std::cerr << "  --quantize: Enable INT8 quantization\n";
    std::cerr << "  --no-quantize: Disable INT8 quantization\n";
    return 1;
  }

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const std::string checkpoint_path = argv[1];
  const std::string tokenizer_path = argv[2];

  DeviceType device = DeviceType::CPU;
  bool enable_quantize = false;
  bool quantize_explicitly_set = false;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

    if (arg == "cuda") {
      device = DeviceType::CUDA;
    } else if (arg == "cpu") {
      device = DeviceType::CPU;
    } else if (arg == "--quantize") {
      enable_quantize = true;
      quantize_explicitly_set = true;
    } else if (arg == "--no-quantize") {
      enable_quantize = false;
      quantize_explicitly_set = true;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      return 1;
    }
  }

  if (!quantize_explicitly_set && device == DeviceType::CUDA) {
    enable_quantize = true;
  }

  std::cout << "PhotonInfer - LLaMA Inference Demo\n";
  std::cout << "===================================\n";
  std::cout << "Device: " << (device == DeviceType::CPU ? "CPU" : "CUDA") << "\n";
  std::cout << "Quantization: " << (enable_quantize ? "INT8" : "FP32") << "\n\n";

  std::cout << "Loading tokenizer from: " << tokenizer_path << "\n";
  auto tokenizer_result = TikTokenizer::load(tokenizer_path);
  if (!tokenizer_result) {
    std::cerr << "Error loading tokenizer: " << tokenizer_result.error().message() << "\n";
    return 1;
  }
  TikTokenizer tokenizer = std::move(tokenizer_result.value());
  std::cout << "  Vocab size: " << tokenizer.vocab_size() << "\n\n";

  std::cout << "Loading checkpoint from: " << checkpoint_path << "\n";
  auto loader_result = CheckpointLoader::open(checkpoint_path);
  if (!loader_result) {
    std::cerr << "Error loading checkpoint: " << loader_result.error().message() << "\n";
    return 1;
  }
  auto loader = std::move(loader_result.value());

  const auto& header = loader->header();
  std::cout << "  Layers: " << header.n_layers << "\n";
  std::cout << "  Dimension: " << header.dim << "\n\n";

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
  config.device = device;
  config.compute_derived();

  std::cout << "Creating model...\n";
  LLaMAModel model(config);

  std::cout << "Loading weights...\n";
  auto load_result = loader->load_weights(model);
  if (!load_result) {
    std::cerr << "Error loading weights: " << load_result.error().message() << "\n";
    return 1;
  }

  auto init_result = model.init();
  if (!init_result) {
    std::cerr << "Error initializing model: " << init_result.error().message() << "\n";
    return 1;
  }

  if (enable_quantize) {
    auto quant_result = model.quantize_weights(128);
    if (!quant_result) {
      std::cerr << "Error quantizing model: " << quant_result.error().message() << "\n";
      return 1;
    }
  }

  std::cout << "\nModel ready!\n\n";

  // Initialize generation state with logits buffer
  GenerationState state;
  auto logits_result = Tensor::create({config.vocab_size}, DataType::Float32, DeviceType::CPU);
  if (!logits_result) {
    std::cerr << "Error: Failed to create logits buffer\n";
    return 1;
  }
  state.logits = std::move(logits_result.value());

  // First turn: Tell the model your name
  std::cout << "=== Turn 1 ===\n";
  const std::string prompt1 = "My name is Lummy. Remember that.";
  std::cout << "User: " << prompt1 << "\n";
  std::cout << "Assistant: ";

  auto start1 = std::chrono::steady_clock::now();
  i32 tokens1 = generate_turn(model, tokenizer, state, prompt1, 128, true);
  auto end1 = std::chrono::steady_clock::now();

  auto duration1 = std::chrono::duration<double>(end1 - start1).count();
  std::cout << "Generated " << tokens1 << " tokens in " << duration1 << "s ("
            << (static_cast<double>(tokens1) / duration1) << " tokens/s)\n\n";

  // Second turn: Ask the model to recall your name
  std::cout << "=== Turn 2 ===\n";
  const std::string prompt2 = "What is my name?";
  std::cout << "User: " << prompt2 << "\n";
  std::cout << "Assistant: ";

  auto start2 = std::chrono::steady_clock::now();
  i32 tokens2 = generate_turn(model, tokenizer, state, prompt2, 128, true);
  auto end2 = std::chrono::steady_clock::now();

  auto duration2 = std::chrono::duration<double>(end2 - start2).count();
  std::cout << "Generated " << tokens2 << " tokens in " << duration2 << "s ("
            << (static_cast<double>(tokens2) / duration2) << " tokens/s)\n\n";

  std::cout << "=== Context Test Complete ===\n";
  std::cout << "Total context length: " << state.current_pos << " tokens\n";

  return 0;
}
