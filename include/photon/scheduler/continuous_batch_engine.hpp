/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file continuous_batch_engine.hpp
 * @brief Continuous batching inference engine
 *
 * Integrates scheduler + model for token-level continuous batching.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include "photon/core/tensor.hpp"
#include "photon/arch/llama_model.hpp"
#include "photon/io/tokenizer.hpp"
#include "photon/runtime/kv_cache_manager.hpp"
#include "photon/scheduler/continuous_batch_scheduler.hpp"
#include "photon/scheduler/inference_request.hpp"

#ifdef PHOTON_USE_CUDA
#include "photon/ops/kernels/cuda/sampling_kernel.cuh"
#endif

namespace photon {
namespace scheduler {

/**
 * @brief Continuous batching inference engine
 *
 * Main execution loop:
 * 1. scheduler.schedule_next_batch() -> get batch
 * 2. Prepare inputs (tokens, positions, seq_ids)
 * 3. model.forward_batched()
 * 4. Sample tokens
 * 5. Update requests and scheduler
 * 6. Repeat until no work
 */
class ContinuousBatchEngine {
 public:
  /**
   * @brief Create engine
   *
   * @param model LLaMA model (must have paged cache initialized)
   * @param tokenizer Tokenizer
   * @param max_batch_size Maximum batch size
   */
  ContinuousBatchEngine(
      model::LLaMAModel& model,
      const model::TikTokenizer& tokenizer,
      i32 max_batch_size)
      : model_(model),
        tokenizer_(tokenizer),
        max_batch_size_(max_batch_size),
        scheduler_(max_batch_size, 32, SchedulingPolicy::FCFS) {

    // Allocate GPU buffers for sampling
#ifdef PHOTON_USE_CUDA
    cudaMalloc(&sampled_tokens_gpu_, max_batch_size * sizeof(i32));
#endif
  }

  ~ContinuousBatchEngine() {
#ifdef PHOTON_USE_CUDA
    if (sampled_tokens_gpu_) {
      cudaFree(sampled_tokens_gpu_);
    }
#endif
  }

  /**
   * @brief Add a new inference request
   *
   * @param prompt Input prompt
   * @param max_new_tokens Maximum tokens to generate
   * @return Request ID
   */
  i64 add_request(const std::string& prompt, i32 max_new_tokens) {
    // Tokenize prompt
    auto tokens = tokenizer_.encode(prompt);
    tokens.insert(tokens.begin(), tokenizer_.bos_id());

    // Add to scheduler
    i64 request_id = scheduler_.add_request(prompt, tokens, max_new_tokens);

    // Allocate KV cache
    auto* cache_mgr = model_.paged_cache_manager();
    i32 total_tokens = static_cast<i32>(tokens.size()) + max_new_tokens + 10;
    auto alloc_result = cache_mgr->allocate_sequence(static_cast<i32>(request_id), total_tokens);
    if (!alloc_result) {
      LOG(ERROR) << "Failed to allocate cache for request " << request_id;
    }

    return request_id;
  }

  /**
   * @brief Run inference until all requests complete
   *
   * @param print_progress Whether to print progress
   * @return Error if failed
   */
  Result<void> run_until_complete(bool print_progress = false) {
    i32 step = 0;

    while (scheduler_.has_work()) {
      // Schedule next batch
      auto batch = scheduler_.schedule_next_batch();
      if (batch.empty()) {
        break;
      }

      if (print_progress && step % 10 == 0) {
        auto stats = scheduler_.get_stats();
        LOG(INFO) << "Step " << step << ": Running=" << stats.num_running
                  << " Waiting=" << stats.num_waiting
                  << " Finished=" << stats.num_finished;
      }

      // Execute batch
      auto result = execute_batch_step(batch);
      if (!result) {
        return Err<void>(result.error());
      }

      step++;
    }

    if (print_progress) {
      auto stats = scheduler_.get_stats();
      LOG(INFO) << "Complete! Total steps: " << step
                << " Total requests: " << stats.total_requests;
    }

    return Ok();
  }

  /**
   * @brief Get result for a request
   *
   * @param request_id Request ID
   * @return Generated text, or empty if not finished
   */
  std::string get_result(i64 request_id) {
    auto request = scheduler_.get_request(request_id);
    if (!request || !request->is_finished()) {
      return "";
    }

    std::string result;
    for (i32 token : request->generated_tokens()) {
      result += tokenizer_.decode_token(token);
    }
    return result;
  }

  /**
   * @brief Get request object
   */
  InferenceRequestPtr get_request(i64 request_id) {
    return scheduler_.get_request(request_id);
  }

  /**
   * @brief Get scheduler stats
   */
  auto get_stats() const {
    return scheduler_.get_stats();
  }

 private:
  /**
   * @brief Execute one step for a batch
   *
   * @param batch Batch of requests
   * @return Error if failed
   */
  Result<void> execute_batch_step(const ScheduledBatch& batch) {
    i32 batch_size = batch.batch_size();

    // Prepare inputs
    std::vector<i32> tokens_batch(batch_size);
    std::vector<i32> positions_batch(batch_size);
    std::vector<i32> seq_ids(batch_size);

    for (i32 i = 0; i < batch_size; ++i) {
      auto& req = batch.requests[i];
      tokens_batch[i] = req->next_token();
      positions_batch[i] = req->current_position();
      seq_ids[i] = static_cast<i32>(req->request_id());
    }

    // Allocate logits tensor
    auto logits_result = Tensor::create(
        {batch_size, model_.config().vocab_size},
        DataType::Float32,
        DeviceType::CUDA);
    if (!logits_result) {
      return Err<void>(logits_result.error());
    }
    auto logits = std::move(logits_result.value());

    // Forward pass
    auto forward_result = model_.forward_batched(tokens_batch, positions_batch, seq_ids, logits);
    if (!forward_result) {
      return Err<void>(forward_result.error());
    }

#ifdef PHOTON_USE_CUDA
    cudaDeviceSynchronize();

    // GPU sampling
    std::vector<i32> sampled_tokens(batch_size);
    auto sampling_result = kernels::cuda::argmax_sampling_launch(
        logits.ptr<f32>(),
        sampled_tokens_gpu_,
        batch_size,
        model_.config().vocab_size,
        nullptr);

    if (!sampling_result) {
      return Err<void>(sampling_result.error());
    }

    cudaMemcpy(sampled_tokens.data(), sampled_tokens_gpu_,
               batch_size * sizeof(i32), cudaMemcpyDeviceToHost);

    // Update requests
    std::vector<i64> finished_ids;
    for (i32 i = 0; i < batch_size; ++i) {
      auto& req = batch.requests[i];

      // Add computed tokens (1 for decode, or chunk for prefill)
      if (req->is_prefill()) {
        req->add_computed_tokens(1);
        // Don't generate during prefill
      } else {
        // Decode: add generated token
        bool should_continue = req->add_token(sampled_tokens[i], tokenizer_.eos_id());
        if (!should_continue) {
          finished_ids.push_back(req->request_id());

          // Free KV cache
          auto* cache_mgr = model_.paged_cache_manager();
          cache_mgr->free_sequence(static_cast<i32>(req->request_id()));
        }
      }
    }

    // Update scheduler
    scheduler_.update_after_step(finished_ids);
#endif

    return Ok();
  }

  model::LLaMAModel& model_;
  const model::TikTokenizer& tokenizer_;
  i32 max_batch_size_;
  ContinuousBatchScheduler scheduler_;

#ifdef PHOTON_USE_CUDA
  i32* sampled_tokens_gpu_ = nullptr;
#endif
};

}  // namespace scheduler
}  // namespace photon
