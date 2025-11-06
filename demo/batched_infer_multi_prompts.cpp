/**
 * @file batched_infer_multi_prompts.cpp
 * @brief Batched inference with multiple different prompts
 *
 * Test batched inference with different prompts running in parallel,
 * showing both correctness and performance.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <glog/logging.h>

#include "photon/model/llama_model.hpp"
#include "photon/model/checkpoint.hpp"
#include "photon/model/tokenizer.hpp"
#include "photon/model/kv_cache_manager.hpp"

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#include "photon/ops/kernels/cuda/sampling_kernel.cuh"
#endif

using namespace photon;
using namespace photon::model;

/**
 * @brief Simple CPU argmax (greedy sampling)
 */
i32 argmax_cpu(const f32* logits, i32 vocab_size) {
  i32 max_idx = 0;
  f32 max_val = logits[0];
  for (i32 i = 1; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }
  return max_idx;
}

/**
 * @brief Sample next token from logits using temperature sampling
 */
i32 sample_token(const f32* logits, i32 vocab_size, f32 temperature = 0.8f) {
  // Use greedy decoding for speed
  if (temperature <= 0.0f) {
    return argmax_cpu(logits, vocab_size);
  }

  std::vector<f32> probs(vocab_size);
  f32 max_logit = *std::max_element(logits, logits + vocab_size);

  f32 sum_exp = 0.0f;
  for (i32 i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum_exp += probs[i];
  }

  for (i32 i = 0; i < vocab_size; ++i) {
    probs[i] /= sum_exp;
  }

  f32 rand_val = static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX);
  f32 cumulative = 0.0f;
  for (i32 i = 0; i < vocab_size; ++i) {
    cumulative += probs[i];
    if (rand_val <= cumulative) {
      return i;
    }
  }

  return vocab_size - 1;
}

/**
 * @brief Run batched inference with multiple different prompts
 */
Result<void> batched_inference_multi_prompts(
    LLaMAModel& model,
    const TikTokenizer& tokenizer,
    const std::vector<std::string>& prompts,
    i32 max_new_tokens,
    bool print_output = true) {

  auto start = std::chrono::high_resolution_clock::now();
  i32 batch_size = static_cast<i32>(prompts.size());

  if (print_output) {
    LOG(INFO) << "\n========================================";
    LOG(INFO) << "Batched Inference with Multiple Prompts";
    LOG(INFO) << "========================================";
    LOG(INFO) << "Batch size: " << batch_size;
    LOG(INFO) << "Max new tokens: " << max_new_tokens << "\n";
  }

  // Encode all prompts
  std::vector<std::vector<i32>> prompt_tokens_list(batch_size);
  i32 max_prompt_len = 0;

  for (i32 i = 0; i < batch_size; ++i) {
    auto tokens = tokenizer.encode(prompts[i]);
    if (tokens.empty()) {
      return Err<void>(ErrorCode::InvalidArgument, "Failed to encode prompt " + std::to_string(i));
    }
    tokens.insert(tokens.begin(), tokenizer.bos_id());
    prompt_tokens_list[i] = std::move(tokens);
    max_prompt_len = std::max(max_prompt_len, static_cast<i32>(prompt_tokens_list[i].size()));

    if (print_output) {
      LOG(INFO) << "Prompt " << i << ": \"" << prompts[i] << "\"";
      LOG(INFO) << "  Tokens (" << prompt_tokens_list[i].size() << "): ";
      for (auto token : prompt_tokens_list[i]) {
        std::cout << token << " ";
      }
      std::cout << "\n";
    }
  }

  // Get cache manager
  auto* cache_mgr = model.cache_manager();
  if (!cache_mgr) {
    return Err<void>(ErrorCode::InvalidOperator, "Cache manager not initialized");
  }

  // Allocate sequences
  std::vector<i32> seq_ids(batch_size);
  i32 total_tokens_needed = max_prompt_len + max_new_tokens + 10;
  for (i32 i = 0; i < batch_size; ++i) {
    seq_ids[i] = i;
    auto alloc_result = cache_mgr->allocate_sequence(i, total_tokens_needed);
    if (!alloc_result) {
      return Err<void>(alloc_result.error());
    }
  }

  // Allocate output tensors
  auto logits_cuda_result = Tensor::create({batch_size, model.config().vocab_size},
                                           DataType::Float32,
                                           DeviceType::CUDA);
  if (!logits_cuda_result) {
    return Err<void>(logits_cuda_result.error());
  }
  auto logits_cuda = std::move(logits_cuda_result.value());

  // Prepare batch inputs
  std::vector<i32> tokens_batch(batch_size);
  std::vector<i32> positions_batch(batch_size);
  std::vector<std::vector<i32>> generated_sequences(batch_size);
  std::vector<i32> current_positions(batch_size, 0);  // Track position per sequence
  std::vector<bool> finished(batch_size, false);      // Track if sequence finished

  if (print_output) {
    LOG(INFO) << "\nGenerating (all " << batch_size << " sequences in parallel):\n";
    for (i32 i = 0; i < batch_size; ++i) {
      std::cout << "Seq " << i << ": " << prompts[i] << " -> " << std::flush;
    }
    std::cout << "\n\n";
  }

  i32 total_tokens_generated = 0;

  // Performance tracking
  double total_forward_time = 0.0;
  double total_copy_time = 0.0;
  double total_sample_time = 0.0;
  i32 num_forwards = 0;
  i32 num_copies = 0;

  // ========== PREFILL PHASE: Process all prompt tokens at once ==========
  LOG(INFO) << "\n========== PREFILL PHASE ==========";
  LOG(INFO) << "Max prompt len: " << max_prompt_len;
  auto prefill_start_time = std::chrono::high_resolution_clock::now();

  for (i32 pos = 0; pos < max_prompt_len; ++pos) {
    // LOG(INFO) << "Prefill position " << pos << "/" << max_prompt_len;
    // Check if any sequence still has tokens to process at this position
    bool has_work = false;
    for (i32 i = 0; i < batch_size; ++i) {
      if (pos < static_cast<i32>(prompt_tokens_list[i].size())) {
        tokens_batch[i] = prompt_tokens_list[i][pos];
        positions_batch[i] = pos;
        current_positions[i] = pos;
        has_work = true;
      } else {
        // Pad with EOS if prompt is shorter
        tokens_batch[i] = tokenizer.eos_id();
        positions_batch[i] = pos;
      }
    }

    if (!has_work) break;

    // Forward pass
    auto forward_start = std::chrono::high_resolution_clock::now();
    auto result = model.forward_batched(tokens_batch, positions_batch, seq_ids, logits_cuda);

    if (!result) {
      LOG(ERROR) << "Prefill failed at position " << pos << ": " << result.error().message();
      // Free sequences
      for (i32 i = 0; i < batch_size; ++i) {
        cache_mgr->free_sequence(i);
      }
      return Err<void>(result.error());
    }

#ifdef PHOTON_USE_CUDA
    // Synchronize to get accurate forward timing
    cudaDeviceSynchronize();
#endif
    auto forward_end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double>(forward_end - forward_start).count();
    num_forwards++;
  }

  auto prefill_end_time = std::chrono::high_resolution_clock::now();
  double prefill_time = std::chrono::duration<double>(prefill_end_time - prefill_start_time).count();
  i32 num_prefill_forwards = num_forwards;

  LOG(INFO) << "Prefill complete: " << num_prefill_forwards << " forwards, "
            << prefill_time << "s";

  // ========== DECODE PHASE: Generate tokens one by one ==========
  LOG(INFO) << "\n========== DECODE PHASE ==========";

#ifdef PHOTON_USE_CUDA
  // Pre-allocate GPU buffer for sampled tokens (reused across decode loop)
  i32* sampled_tokens_gpu;
  cudaMalloc(&sampled_tokens_gpu, batch_size * sizeof(i32));
  std::vector<i32> sampled_tokens(batch_size);
#endif

  // Generation loop
  for (i32 step = 0; step < max_new_tokens; ++step) {
    // Check if all sequences finished
    bool all_finished = true;
    for (i32 i = 0; i < batch_size; ++i) {
      if (!finished[i]) {
        all_finished = false;
        break;
      }
    }
    if (all_finished) break;

    // Prepare batch inputs for decode phase
    for (i32 i = 0; i < batch_size; ++i) {
      if (finished[i]) {
        tokens_batch[i] = tokenizer.eos_id();
        positions_batch[i] = current_positions[i];
      } else {
        // Use last generated token, or last prompt token for first decode
        if (generated_sequences[i].empty()) {
          // First decode step: use last prompt token
          tokens_batch[i] = prompt_tokens_list[i].back();
        } else {
          // Subsequent decode steps: use last generated token
          tokens_batch[i] = generated_sequences[i].back();
        }
        current_positions[i]++;
        positions_batch[i] = current_positions[i];
      }
    }

    // Forward pass
    auto forward_start = std::chrono::high_resolution_clock::now();
    auto result = model.forward_batched(tokens_batch, positions_batch, seq_ids, logits_cuda);

    if (!result) {
      LOG(ERROR) << "Decode failed at step " << step << ": " << result.error().message();
      // Free sequences
      for (i32 i = 0; i < batch_size; ++i) {
        cache_mgr->free_sequence(i);
      }
      return Err<void>(result.error());
    }

#ifdef PHOTON_USE_CUDA
    // Synchronize to get accurate forward timing
    cudaDeviceSynchronize();
#endif
    auto forward_end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double>(forward_end - forward_start).count();
    num_forwards++;

    // GPU-based argmax sampling (no logits copy needed!)
    auto sample_start = std::chrono::high_resolution_clock::now();

#ifdef PHOTON_USE_CUDA  // Use GPU sampling
    // Launch GPU argmax kernel (using pre-allocated buffer)
    auto sampling_result = photon::kernels::cuda::argmax_sampling_launch(
        logits_cuda.ptr<f32>(),
        sampled_tokens_gpu,
        batch_size,
        model.config().vocab_size,
        nullptr);

    if (!sampling_result) {
      LOG(ERROR) << "GPU sampling failed: " << sampling_result.error().message();
      cudaFree(sampled_tokens_gpu);
      return Err<void>(sampling_result.error());
    }

    // Copy only the sampled tokens (batch_size integers) from GPU to CPU
    auto copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(sampled_tokens.data(), sampled_tokens_gpu,
               batch_size * sizeof(i32), cudaMemcpyDeviceToHost);
    auto copy_end = std::chrono::high_resolution_clock::now();
    total_copy_time += std::chrono::duration<double>(copy_end - copy_start).count();
    num_copies++;

    // Update sequences with sampled tokens
    for (i32 i = 0; i < batch_size; ++i) {
      if (finished[i]) continue;

      i32 next_token = sampled_tokens[i];
      generated_sequences[i].push_back(next_token);
      total_tokens_generated++;

      // Check for EOS
      if (next_token == tokenizer.eos_id() ||
          static_cast<i32>(generated_sequences[i].size()) >= max_new_tokens) {
        finished[i] = true;
      }
    }
#else
    // CPU fallback: copy logits and sample
    auto logits_cpu_result = logits_cuda.to(DeviceType::CPU);
    if (!logits_cpu_result) {
      return Err<void>(logits_cpu_result.error());
    }
    auto logits_cpu = std::move(logits_cpu_result.value());
    const f32* logits_ptr = logits_cpu.ptr<f32>();

    for (i32 i = 0; i < batch_size; ++i) {
      if (finished[i]) continue;

      const f32* seq_logits = logits_ptr + i * model.config().vocab_size;
      i32 next_token = sample_token(seq_logits, model.config().vocab_size, 0.8f);

      generated_sequences[i].push_back(next_token);
      total_tokens_generated++;

      // Check for EOS
      if (next_token == tokenizer.eos_id() ||
          static_cast<i32>(generated_sequences[i].size()) >= max_new_tokens) {
        finished[i] = true;
      }
    }
#endif

    auto sample_end = std::chrono::high_resolution_clock::now();
    total_sample_time += std::chrono::duration<double>(sample_end - sample_start).count();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();

#ifdef PHOTON_USE_CUDA
  // Free GPU sampling buffer
  cudaFree(sampled_tokens_gpu);
#endif

  // Print results
  if (print_output) {
    LOG(INFO) << "\n========================================";
    LOG(INFO) << "Generated Results:";
    LOG(INFO) << "========================================\n";

    for (i32 i = 0; i < batch_size; ++i) {
      std::string generated_text;
      for (i32 token : generated_sequences[i]) {
        generated_text += tokenizer.decode_token(token);
      }
      LOG(INFO) << "Prompt " << i << ": \"" << prompts[i] << "\"";
      LOG(INFO) << "Generated (" << generated_sequences[i].size() << " tokens): " << generated_text << "\n";
    }

    LOG(INFO) << "========================================";
    LOG(INFO) << "Performance Statistics:";
    LOG(INFO) << "========================================";
    LOG(INFO) << "Total tokens generated: " << total_tokens_generated;
    LOG(INFO) << "Time: " << duration << " seconds";
    LOG(INFO) << "Throughput: " << (static_cast<double>(total_tokens_generated) / duration) << " tokens/s";
    LOG(INFO) << "Average per sequence: " << (static_cast<double>(total_tokens_generated) / batch_size) << " tokens";
    LOG(INFO) << "\nTime Breakdown:";
    LOG(INFO) << "  Forward passes: " << num_forwards << " calls, " << total_forward_time << "s ("
              << (total_forward_time/duration*100) << "%)";
    LOG(INFO) << "  GPU→CPU copies: " << num_copies << " calls, " << total_copy_time << "s ("
              << (total_copy_time/duration*100) << "%)";
    LOG(INFO) << "  Sampling: " << total_sample_time << "s ("
              << (total_sample_time/duration*100) << "%)\n";
  }

  // Free sequences
  for (i32 i = 0; i < batch_size; ++i) {
    auto free_result = cache_mgr->free_sequence(i);
    if (!free_result) {
      LOG(WARNING) << "Failed to free sequence " << i << ": " << free_result.error().message();
    }
  }

  return Ok();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " <model_path> <tokenizer_path>";
    LOG(INFO) << "Example: " << argv[0]
              << " ~/.llama/checkpoints/Llama3.2-1B-Instruct/model.bin"
              << " ~/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model";
    return 1;
  }

  std::string model_path = argv[1];
  std::string tokenizer_path = argv[2];

  LOG(INFO) << "========================================";
  LOG(INFO) << "Batched Multi-Prompt Inference";
  LOG(INFO) << "========================================\n";

  // Load tokenizer
  LOG(INFO) << "[1/4] Loading tokenizer...";
  auto tokenizer_result = TikTokenizer::load(tokenizer_path);
  if (!tokenizer_result) {
    LOG(ERROR) << "Failed to load tokenizer: " << tokenizer_result.error().message();
    return 1;
  }
  TikTokenizer tokenizer = std::move(tokenizer_result.value());
  LOG(INFO) << "      Vocab size: " << tokenizer.vocab_size();

  // Load checkpoint
  LOG(INFO) << "\n[2/4] Loading checkpoint...";
  auto loader_result = CheckpointLoader::open(model_path);
  if (!loader_result) {
    LOG(ERROR) << "Failed to load checkpoint: " << loader_result.error().message();
    return 1;
  }
  auto loader = std::move(loader_result.value());

  const auto& header = loader->header();
  LOG(INFO) << "      Model: Llama 3.2 1B";
  LOG(INFO) << "      Dimension: " << header.dim;
  LOG(INFO) << "      Layers: " << header.n_layers;

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
  config.device = DeviceType::CUDA;
  config.compute_derived();

  // Create model
  LLaMAModel model(config);

  // Load weights
  LOG(INFO) << "\n[3/4] Loading weights and initializing...";
  auto load_result = loader->load_weights(model);
  if (!load_result) {
    LOG(ERROR) << "Failed to load weights: " << load_result.error().message();
    return 1;
  }

  auto init_result = model.init();
  if (!init_result) {
    LOG(ERROR) << "Failed to initialize model: " << init_result.error().message();
    return 1;
  }

  // Enable INT8 quantization
  LOG(INFO) << "      Enabling INT8 quantization...";
  auto quant_result = model.quantize_weights();
  if (!quant_result) {
    LOG(ERROR) << "Failed to quantize: " << quant_result.error().message();
    return 1;
  }

  // Initialize paged KV cache
  LOG(INFO) << "\n[4/4] Initializing paged KV cache...";
  i32 max_sequences = 32;  // Maximum concurrent sequences
  i32 max_seq_len = 256;   // Maximum tokens per sequence (prompt + generation)
  auto cache_result = model.init_paged_cache(max_sequences, max_seq_len);
  if (!cache_result) {
    LOG(ERROR) << "Failed to initialize paged cache: " << cache_result.error().message();
    return 1;
  }
  LOG(INFO) << "      Cache: " << max_sequences << " sequences × " << max_seq_len
            << " tokens/seq";
  LOG(INFO) << "      Model ready!\n";

  // Test with multiple prompts for true batched inference
  std::vector<std::string> test_prompts = {
    "What is your name?",
    "Tell me a story about a cat.",
    "How does photosynthesis work?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What is machine learning?",
    "Describe our solar system."
  };

  i32 max_new_tokens = 100;  // Generate 100 tokens per sequence

  auto result = batched_inference_multi_prompts(model, tokenizer, test_prompts,
                                                max_new_tokens, true);
  if (!result) {
    LOG(ERROR) << "Batched inference failed: " << result.error().message();
    return 1;
  }

  LOG(INFO) << "\n✓ Test complete!";

  return 0;
}
