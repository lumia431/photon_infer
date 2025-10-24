/**
 * @file debug_first_token.cpp
 * @brief Compare CPU vs CUDA inference for the first token
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include "photon/model/llama_model.hpp"
#include "photon/model/checkpoint.hpp"
#include "photon/model/tokenizer.hpp"

using namespace photon;
using namespace photon::model;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <model.bin> <tokenizer.model>\n";
    return 1;
  }

  const char* model_path = argv[1];
  const char* tokenizer_path = argv[2];

  std::cout << "=== Debugging First Token: CPU vs CUDA ===\n\n";

  // Load tokenizer
  std::cout << "Loading tokenizer...\n";
  auto tokenizer_result = TikTokenizer::load(tokenizer_path);
  if (!tokenizer_result) {
    std::cerr << "Failed to load tokenizer: " << tokenizer_result.error().message() << "\n";
    return 1;
  }
  TikTokenizer tokenizer = std::move(tokenizer_result.value());

  // Encode prompt
  const std::string prompt = "What is your name?";
  auto tokens = tokenizer.encode(prompt);
  std::cout << "Prompt: \"" << prompt << "\"\n";
  std::cout << "Tokens: ";
  for (auto t : tokens) std::cout << t << " ";
  std::cout << "\n\n";

  // Load checkpoint
  std::cout << "Loading checkpoint...\n";
  auto loader_result = CheckpointLoader::open(model_path);
  if (!loader_result) {
    std::cerr << "Failed to load checkpoint: " << loader_result.error().message() << "\n";
    return 1;
  }
  auto checkpoint = std::move(loader_result.value());
  const auto& config = checkpoint->header();

  std::cout << "Model config:\n";
  std::cout << "  Dim: " << config.dim << "\n";
  std::cout << "  Layers: " << config.n_layers << "\n";
  std::cout << "  Vocab: " << config.vocab_size << "\n\n";

  // Create model config
  TransformerConfig model_config;
  model_config.dim = config.dim;
  model_config.hidden_dim = config.hidden_dim;
  model_config.n_layers = config.n_layers;
  model_config.n_heads = config.n_heads;
  model_config.n_kv_heads = config.n_kv_heads;
  model_config.vocab_size = config.vocab_size;
  model_config.seq_len = config.seq_len;
  model_config.head_size = config.dim / config.n_heads;
  model_config.norm_eps = 1e-5f;  // Default RMSNorm epsilon

  // ==================================================================
  // CPU Model
  // ==================================================================
  std::cout << "Creating CPU model...\n";
  model_config.device = DeviceType::CPU;
  LLaMAModel model_cpu(model_config);

  std::cout << "Loading weights to CPU model...\n";
  auto load_cpu_result = checkpoint->load_weights(model_cpu);
  if (!load_cpu_result) {
    std::cerr << "Failed to load CPU weights: " << load_cpu_result.error().message() << "\n";
    return 1;
  }

  std::cout << "Initializing CPU model...\n";
  auto init_cpu_result = model_cpu.init();
  if (!init_cpu_result) {
    std::cerr << "Failed to init CPU model: " << init_cpu_result.error().message() << "\n";
    return 1;
  }

  // ==================================================================
  // CUDA Model
  // ==================================================================
  std::cout << "Creating CUDA model...\n";
  model_config.device = DeviceType::CUDA;
  LLaMAModel model_cuda(model_config);

  std::cout << "Loading weights to CUDA model...\n";
  auto load_cuda_result = checkpoint->load_weights(model_cuda);
  if (!load_cuda_result) {
    std::cerr << "Failed to load CUDA weights: " << load_cuda_result.error().message() << "\n";
    return 1;
  }

  std::cout << "Initializing CUDA model...\n";
  auto init_cuda_result = model_cuda.init();
  if (!init_cuda_result) {
    std::cerr << "Failed to init CUDA model: " << init_cuda_result.error().message() << "\n";
    return 1;
  }

  std::cout << "\n=== Processing prompt tokens ===\n";

  // Create logits buffers
  std::vector<int32_t> dims = {static_cast<int32_t>(config.vocab_size)};
  auto logits_cpu = Tensor::create(dims, DataType::Float32, DeviceType::CPU).value();
  auto logits_cuda_on_cpu = Tensor::create(dims, DataType::Float32, DeviceType::CPU).value();

  // Process all prompt tokens
  for (usize i = 0; i < tokens.size(); ++i) {
    i32 token = tokens[i];
    i32 pos = static_cast<i32>(i);

    std::cout << "\nPosition " << pos << ", Token " << token << ":\n";

    // CPU forward
    auto cpu_fwd = model_cpu.forward(token, pos, logits_cpu);
    if (!cpu_fwd) {
      std::cerr << "CPU forward failed: " << cpu_fwd.error().message() << "\n";
      return 1;
    }

    // CUDA forward
    auto cuda_fwd = model_cuda.forward(token, pos, logits_cuda_on_cpu);
    if (!cuda_fwd) {
      std::cerr << "CUDA forward failed: " << cuda_fwd.error().message() << "\n";
      return 1;
    }

    // Compare logits
    auto cpu_map = logits_cpu.vector_map<f32>();
    auto cuda_map = logits_cuda_on_cpu.vector_map<f32>();

    // Find top-5 tokens for both
    std::vector<std::pair<float, i32>> cpu_top, cuda_top;
    for (i32 j = 0; j < config.vocab_size; ++j) {
      cpu_top.push_back(std::make_pair(cpu_map[j], j));
      cuda_top.push_back(std::make_pair(cuda_map[j], j));
    }

    std::sort(cpu_top.begin(), cpu_top.end(), std::greater<>());
    std::sort(cuda_top.begin(), cuda_top.end(), std::greater<>());

    std::cout << "  CPU  top-5: ";
    for (int k = 0; k < 5; ++k) {
      std::cout << "(" << cpu_top[k].second << "," << cpu_top[k].first << ") ";
    }
    std::cout << "\n";

    std::cout << "  CUDA top-5: ";
    for (int k = 0; k < 5; ++k) {
      std::cout << "(" << cuda_top[k].second << "," << cuda_top[k].first << ") ";
    }
    std::cout << "\n";

    // Compare if top-1 matches
    if (cpu_top[0].second != cuda_top[0].second) {
      std::cout << "  ✗ TOP-1 MISMATCH! CPU=" << cpu_top[0].second
                << " CUDA=" << cuda_top[0].second << "\n";
    } else {
      std::cout << "  ✓ Top-1 matches: " << cpu_top[0].second << "\n";
    }

    // Compute stats
    f32 max_diff = 0.0f;
    f32 sum_sq_diff = 0.0f;
    int mismatches = 0;
    for (i32 j = 0; j < config.vocab_size; ++j) {
      f32 diff = std::abs(cpu_map[j] - cuda_map[j]);
      sum_sq_diff += diff * diff;
      if (diff > max_diff) max_diff = diff;
      if (diff > 0.1f) mismatches++;
    }

    f32 rmse = std::sqrt(sum_sq_diff / config.vocab_size);
    std::cout << "  Logits: max_diff=" << max_diff << ", RMSE=" << rmse
              << ", mismatches(>0.1)=" << mismatches << "\n";
  }

  std::cout << "\n=== Done ===\n";
  return 0;
}
