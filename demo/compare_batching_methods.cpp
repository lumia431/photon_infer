/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file compare_batching_methods.cpp
 * @brief Comparison between baseline and continuous batching
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <glog/logging.h>

#include "photon/scheduler/continuous_batch_engine.hpp"
#include "photon/arch/llama_model.hpp"
#include "photon/io/checkpoint.hpp"
#include "photon/io/tokenizer.hpp"

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#include "photon/ops/kernels/cuda/sampling_kernel.cuh"
#endif

using namespace photon;
using namespace photon::model;
using namespace photon::scheduler;

std::chrono::high_resolution_clock::time_point g_start_time;

double get_elapsed_time() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now - g_start_time).count();
}

// Request definition
struct TestRequest {
  std::string prompt;
  i32 max_new_tokens;
  double arrival_time;  // seconds after start
  std::string label;
};

// Result tracking
struct RequestResult {
  std::string label;
  double arrival_time;
  double start_time;
  double finish_time;
  i32 tokens_generated;

  double wait_time() const { return start_time - arrival_time; }
  double exec_time() const { return finish_time - start_time; }
  double total_latency() const { return finish_time - arrival_time; }
};

/**
 * @brief Run baseline batch inference (traditional approach)
 */
std::vector<RequestResult> run_baseline(
    LLaMAModel& model,
    const TikTokenizer& tokenizer,
    const std::vector<TestRequest>& requests) {

  LOG(INFO) << "\n========================================";
  LOG(INFO) << "METHOD 1: Baseline (Traditional Batching)";
  LOG(INFO) << "========================================\n";

  g_start_time = std::chrono::high_resolution_clock::now();
  std::vector<RequestResult> results;

  // Group requests into batches by arrival time
  std::vector<std::vector<i32>> batches;  // indices into requests
  std::map<double, std::vector<i32>> arrival_groups;

  for (i32 i = 0; i < static_cast<i32>(requests.size()); ++i) {
    arrival_groups[requests[i].arrival_time].push_back(i);
  }

  // Process each batch sequentially
  for (const auto& [arrival_time, indices] : arrival_groups) {
    // Wait until arrival time
    while (get_elapsed_time() < arrival_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    double batch_start = get_elapsed_time();
    LOG(INFO) << "[t=" << std::fixed << std::setprecision(3)
              << batch_start << "s] Processing batch of " << indices.size() << " requests";

    // Prepare batch
    i32 batch_size = static_cast<i32>(indices.size());
    std::vector<std::vector<i32>> prompt_tokens_list(batch_size);
    std::vector<i32> seq_ids(batch_size);
    std::vector<std::vector<i32>> generated_sequences(batch_size);
    std::vector<bool> finished(batch_size, false);
    i32 max_prompt_len = 0;
    i32 max_new_tokens = 0;

    for (i32 i = 0; i < batch_size; ++i) {
      const auto& req = requests[indices[i]];
      auto tokens = tokenizer.encode(req.prompt);
      tokens.insert(tokens.begin(), tokenizer.bos_id());
      prompt_tokens_list[i] = std::move(tokens);
      max_prompt_len = std::max(max_prompt_len, static_cast<i32>(prompt_tokens_list[i].size()));
      max_new_tokens = std::max(max_new_tokens, req.max_new_tokens);
      seq_ids[i] = indices[i];
    }

    // Allocate KV cache
    auto* cache_mgr = model.paged_cache_manager();
    for (i32 i = 0; i < batch_size; ++i) {
      i32 total_tokens = max_prompt_len + max_new_tokens + 10;
      cache_mgr->allocate_sequence(seq_ids[i], total_tokens);
    }

    // Allocate logits
    auto logits_result = Tensor::create({batch_size, model.config().vocab_size},
                                        DataType::Float32, DeviceType::CUDA);
    if (!logits_result) {
      LOG(ERROR) << "Failed to create logits tensor";
      return results;
    }
    auto logits = std::move(logits_result.value());

    std::vector<i32> tokens_batch(batch_size);
    std::vector<i32> positions_batch(batch_size);
    std::vector<i32> current_positions(batch_size, 0);

    // Prefill phase
    for (i32 pos = 0; pos < max_prompt_len; ++pos) {
      bool has_work = false;
      for (i32 i = 0; i < batch_size; ++i) {
        if (pos < static_cast<i32>(prompt_tokens_list[i].size())) {
          tokens_batch[i] = prompt_tokens_list[i][pos];
          positions_batch[i] = pos;
          current_positions[i] = pos;
          has_work = true;
        } else {
          tokens_batch[i] = tokenizer.eos_id();
          positions_batch[i] = pos;
        }
      }
      if (!has_work) break;

      auto result = model.forward_batched(tokens_batch, positions_batch, seq_ids, logits);
      if (!result) {
        LOG(ERROR) << "Forward failed: " << result.error().message();
        return results;
      }
    }

    // Decode phase
#ifdef PHOTON_USE_CUDA
    i32* sampled_tokens_gpu;
    cudaMalloc(&sampled_tokens_gpu, batch_size * sizeof(i32));
    std::vector<i32> sampled_tokens(batch_size);

    for (i32 step = 0; step < max_new_tokens; ++step) {
      bool all_finished = true;
      for (i32 i = 0; i < batch_size; ++i) {
        if (!finished[i]) {
          all_finished = false;
          break;
        }
      }
      if (all_finished) break;

      // Prepare inputs
      for (i32 i = 0; i < batch_size; ++i) {
        if (finished[i]) {
          tokens_batch[i] = tokenizer.eos_id();
          positions_batch[i] = current_positions[i];
        } else {
          if (generated_sequences[i].empty()) {
            tokens_batch[i] = prompt_tokens_list[i].back();
          } else {
            tokens_batch[i] = generated_sequences[i].back();
          }
          current_positions[i]++;
          positions_batch[i] = current_positions[i];
        }
      }

      // Forward
      auto result = model.forward_batched(tokens_batch, positions_batch, seq_ids, logits);
      if (!result) {
        LOG(ERROR) << "Forward failed: " << result.error().message();
        cudaFree(sampled_tokens_gpu);
        return results;
      }

      cudaDeviceSynchronize();

      // Sample
      auto sampling_result = kernels::cuda::argmax_sampling_launch(
          logits.ptr<f32>(), sampled_tokens_gpu, batch_size,
          model.config().vocab_size, nullptr);
      if (!sampling_result) {
        LOG(ERROR) << "Sampling failed";
        cudaFree(sampled_tokens_gpu);
        return results;
      }

      cudaMemcpy(sampled_tokens.data(), sampled_tokens_gpu,
                 batch_size * sizeof(i32), cudaMemcpyDeviceToHost);

      // Update
      for (i32 i = 0; i < batch_size; ++i) {
        if (finished[i]) continue;
        i32 token = sampled_tokens[i];
        generated_sequences[i].push_back(token);
        if (token == tokenizer.eos_id() ||
            static_cast<i32>(generated_sequences[i].size()) >= requests[indices[i]].max_new_tokens) {
          finished[i] = true;
        }
      }
    }

    cudaFree(sampled_tokens_gpu);
#endif

    double batch_end = get_elapsed_time();

    // Record results
    for (i32 i = 0; i < batch_size; ++i) {
      RequestResult res;
      res.label = requests[indices[i]].label;
      res.arrival_time = arrival_time;
      res.start_time = batch_start;
      res.finish_time = batch_end;
      res.tokens_generated = static_cast<i32>(generated_sequences[i].size());
      results.push_back(res);

      cache_mgr->free_sequence(seq_ids[i]);
    }

    LOG(INFO) << "  Batch completed in " << (batch_end - batch_start) << "s";
  }

  return results;
}

/**
 * @brief Run continuous batching
 */
std::vector<RequestResult> run_continuous_batching(
    LLaMAModel& model,
    const TikTokenizer& tokenizer,
    const std::vector<TestRequest>& requests) {

  LOG(INFO) << "\n========================================";
  LOG(INFO) << "METHOD 2: Continuous Batching";
  LOG(INFO) << "========================================\n";

  g_start_time = std::chrono::high_resolution_clock::now();
  std::vector<RequestResult> results(requests.size());

  ContinuousBatchEngine engine(model, tokenizer, 8);

  // Track request IDs
  std::vector<i64> request_ids(requests.size());

  // Launch inference thread
  std::atomic<bool> done{false};
  std::thread inference_thread([&]() {
    auto result = engine.run_until_complete(false);
    if (!result) {
      LOG(ERROR) << "Inference failed: " << result.error().message();
    }
    done = true;
  });

  // Add requests at their arrival times
  for (i32 i = 0; i < static_cast<i32>(requests.size()); ++i) {
    // Wait until arrival time
    while (get_elapsed_time() < requests[i].arrival_time) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    double arrival = get_elapsed_time();
    i64 id = engine.add_request(requests[i].prompt, requests[i].max_new_tokens);
    request_ids[i] = id;

    LOG(INFO) << "[t=" << std::fixed << std::setprecision(3)
              << arrival << "s] Added request " << i << " (" << requests[i].label << ")";
  }

  // Wait for completion
  inference_thread.join();

  // Collect results
  for (i32 i = 0; i < static_cast<i32>(requests.size()); ++i) {
    auto req = engine.get_request(request_ids[i]);
    if (req) {
      results[i].label = requests[i].label;
      results[i].arrival_time = requests[i].arrival_time;
      results[i].start_time = requests[i].arrival_time;  // Approximation
      results[i].finish_time = requests[i].arrival_time + req->latency_seconds();
      results[i].tokens_generated = req->num_generated();
    }
  }

  return results;
}

/**
 * @brief Print comparison results
 */
void print_comparison(
    const std::vector<RequestResult>& baseline_results,
    const std::vector<RequestResult>& continuous_results) {

  LOG(INFO) << "\n========================================";
  LOG(INFO) << "COMPARISON RESULTS";
  LOG(INFO) << "========================================\n";

  // Per-request comparison
  LOG(INFO) << "Per-Request Latency:";
  LOG(INFO) << std::setw(20) << "Request"
            << std::setw(15) << "Baseline"
            << std::setw(15) << "Continuous"
            << std::setw(15) << "Speedup";

  for (i32 i = 0; i < static_cast<i32>(baseline_results.size()); ++i) {
    double baseline_lat = baseline_results[i].total_latency();
    double continuous_lat = continuous_results[i].total_latency();
    double speedup = baseline_lat / continuous_lat;

    LOG(INFO) << std::setw(20) << baseline_results[i].label
              << std::setw(15) << std::fixed << std::setprecision(3) << baseline_lat << "s"
              << std::setw(15) << continuous_lat << "s"
              << std::setw(15) << std::setprecision(2) << speedup << "x";
  }

  // Summary statistics
  double baseline_total = 0, continuous_total = 0;
  double baseline_max = 0, continuous_max = 0;
  i32 total_tokens = 0;

  for (i32 i = 0; i < static_cast<i32>(baseline_results.size()); ++i) {
    baseline_total += baseline_results[i].total_latency();
    continuous_total += continuous_results[i].total_latency();
    baseline_max = std::max(baseline_max, baseline_results[i].finish_time);
    continuous_max = std::max(continuous_max, continuous_results[i].finish_time);
    total_tokens += baseline_results[i].tokens_generated;
  }

  LOG(INFO) << "\n========================================";
  LOG(INFO) << "Summary:";
  LOG(INFO) << "========================================";
  LOG(INFO) << "Total tokens: " << total_tokens;
  LOG(INFO) << "\nBaseline:";
  LOG(INFO) << "  Total time: " << baseline_max << "s";
  LOG(INFO) << "  Throughput: " << (total_tokens / baseline_max) << " tokens/s";
  LOG(INFO) << "  Avg latency: " << (baseline_total / baseline_results.size()) << "s";

  LOG(INFO) << "\nContinuous Batching:";
  LOG(INFO) << "  Total time: " << continuous_max << "s";
  LOG(INFO) << "  Throughput: " << (total_tokens / continuous_max) << " tokens/s";
  LOG(INFO) << "  Avg latency: " << (continuous_total / continuous_results.size()) << "s";

  LOG(INFO) << "\nImprovement:";
  LOG(INFO) << "  Throughput: " << std::fixed << std::setprecision(2)
            << (baseline_max / continuous_max) << "x";
  LOG(INFO) << "  Avg latency: " << (baseline_total / continuous_total) << "x";
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " <model_path> <tokenizer_path>";
    return 1;
  }

  std::string model_path = argv[1];
  std::string tokenizer_path = argv[2];

  LOG(INFO) << "========================================";
  LOG(INFO) << "Batching Methods Comparison";
  LOG(INFO) << "========================================\n";

  // Load tokenizer
  LOG(INFO) << "Loading tokenizer...";
  auto tokenizer_result = TikTokenizer::load(tokenizer_path);
  if (!tokenizer_result) {
    LOG(ERROR) << "Failed to load tokenizer";
    return 1;
  }
  TikTokenizer tokenizer = std::move(tokenizer_result.value());

  // Load model
  LOG(INFO) << "Loading model...";
  auto loader_result = CheckpointLoader::open(model_path);
  if (!loader_result) {
    LOG(ERROR) << "Failed to load checkpoint";
    return 1;
  }
  auto loader = std::move(loader_result.value());

  const auto& header = loader->header();
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

  LLaMAModel model(config);

  LOG(INFO) << "Loading weights...";
  auto load_result = loader->load_weights(model);
  if (!load_result) {
    LOG(ERROR) << "Failed to load weights";
    return 1;
  }

  auto init_result = model.init();
  if (!init_result) {
    LOG(ERROR) << "Failed to initialize model";
    return 1;
  }

  LOG(INFO) << "Enabling INT8 quantization...";
  auto quant_result = model.quantize_weights();
  if (!quant_result) {
    LOG(ERROR) << "Failed to quantize";
    return 1;
  }

  LOG(INFO) << "Initializing paged KV cache...";
  auto cache_result = model.init_paged_cache(32, 256);
  if (!cache_result) {
    LOG(ERROR) << "Failed to initialize paged cache";
    return 1;
  }

  // Define test requests
  std::vector<TestRequest> test_requests = {
    // Batch 1: Long requests
    {"Write a story about a robot.", 50, 0.0, "Long-1"},
    {"Explain quantum physics.", 50, 0.0, "Long-2"},
    {"Describe the ocean.", 50, 0.0, "Long-3"},
    {"Tell me about space.", 50, 0.0, "Long-4"},

    // Batch 2: Urgent short requests
    {"What is 2+2?", 10, 0.5, "Urgent-1"},
    {"Say hello.", 10, 0.5, "Urgent-2"},
    {"Count to 5.", 10, 0.5, "Urgent-3"},
    {"Name a color.", 10, 0.5, "Urgent-4"},
  };

  // Run baseline
  auto baseline_results = run_baseline(model, tokenizer, test_requests);

  // Small delay between tests
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Run continuous batching
  auto continuous_results = run_continuous_batching(model, tokenizer, test_requests);

  // Print comparison
  print_comparison(baseline_results, continuous_results);

  return 0;
}
