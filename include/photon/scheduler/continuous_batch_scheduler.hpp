/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file continuous_batch_scheduler.hpp
 * @brief Continuous batching scheduler for LLM inference
 *
 * Implements vLLM-style token-level continuous batching:
 * - Two-phase scheduling (RUNNING priority, then WAITING)
 * - Dynamic batch composition at each step
 * - Automatic request completion handling
 */

#pragma once

#include <deque>
#include <vector>
#include <unordered_map>
#include <memory>

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"
#include "photon/scheduler/inference_request.hpp"

namespace photon {
namespace scheduler {

/**
 * @brief Batch of requests to execute together
 */
struct ScheduledBatch {
  std::vector<InferenceRequestPtr> requests;

  i32 batch_size() const {
    return static_cast<i32>(requests.size());
  }

  bool empty() const {
    return requests.empty();
  }
};

/**
 * @brief Scheduling policy
 */
enum class SchedulingPolicy {
  FCFS,      // First-Come-First-Serve
  PRIORITY,  // Priority-based (not yet implemented)
};

/**
 * @brief Continuous batching scheduler with chunked prefill optimization
 *
 * Key algorithm (vLLM-style optimized scheduling):
 *
 * Phase 1: Continue RUNNING requests (both prefill and decode)
 *   - All requests currently being processed must continue
 *   - Prefill requests process in chunks (default 256 tokens)
 *   - Decode requests process 1 token at a time
 *   - This ensures KV cache consistency
 *
 * Phase 2: Add WAITING requests
 *   - Fill remaining batch capacity with new requests
 *   - Prioritize oldest requests (FCFS)
 *   - Can mix prefill and decode in same batch (variable-length batching)
 *
 * Chunked Prefill Benefits:
 *   - Long prompts don't block decode requests
 *   - Better GPU utilization with mixed prefill/decode
 *   - Lower time-to-first-token (TTFT) for short requests
 *
 * After each step:
 *   - Remove finished requests
 *   - Update request states and computed token counts
 */
class ContinuousBatchScheduler {
 public:
  /**
   * @brief Create scheduler
   *
   * @param max_batch_size Maximum number of sequences in a batch
   * @param max_sequences Maximum total sequences (cache capacity)
   * @param chunk_size Prefill chunk size (default 256)
   * @param policy Scheduling policy
   */
  ContinuousBatchScheduler(
      i32 max_batch_size,
      i32 max_sequences,
      i32 chunk_size = 256,
      SchedulingPolicy policy = SchedulingPolicy::FCFS)
      : max_batch_size_(max_batch_size),
        max_sequences_(max_sequences),
        chunk_size_(chunk_size),
        policy_(policy),
        next_seq_id_(0) {}

  /**
   * @brief Add a new inference request to the queue
   *
   * @param prompt Input prompt string
   * @param prompt_tokens Tokenized prompt
   * @param max_new_tokens Maximum tokens to generate
   * @return Request ID
   */
  i64 add_request(
      const std::string& prompt,
      const std::vector<i32>& prompt_tokens,
      i32 max_new_tokens) {
    i64 request_id = next_seq_id_++;
    auto request = std::make_shared<InferenceRequest>(
        request_id, prompt, prompt_tokens, max_new_tokens);

    waiting_queue_.push_back(request);
    request_map_[request_id] = request;

    return request_id;
  }

  /**
   * @brief Schedule next batch for execution
   *
   * This is the core scheduling logic:
   * 1. Continue all RUNNING requests (mandatory)
   * 2. Add WAITING requests to fill batch capacity
   * 3. Respect max_batch_size limit
   *
   * @return Batch to execute, or empty batch if no work
   */
  ScheduledBatch schedule_next_batch() {
    ScheduledBatch batch;

    // Phase 1: Continue RUNNING requests (they have active KV cache)
    for (auto& req : running_requests_) {
      batch.requests.push_back(req);
    }

    // Phase 2: Add WAITING requests to fill capacity
    i32 remaining_capacity = max_batch_size_ - batch.batch_size();
    i32 available_slots = max_sequences_ - num_active_sequences();

    while (remaining_capacity > 0 && available_slots > 0 && !waiting_queue_.empty()) {
      auto req = waiting_queue_.front();
      waiting_queue_.pop_front();

      req->start_running();
      running_requests_.push_back(req);
      batch.requests.push_back(req);

      remaining_capacity--;
      available_slots--;
    }

    return batch;
  }

  /**
   * @brief Update scheduler after batch execution
   *
   * Call this after executing a batch to:
   * - Remove finished requests
   * - Update request states
   *
   * @param finished_ids IDs of requests that finished in this step
   */
  void update_after_step(const std::vector<i64>& finished_ids) {
    // Remove finished requests from running list
    for (i64 id : finished_ids) {
      auto it = std::find_if(
          running_requests_.begin(),
          running_requests_.end(),
          [id](const auto& req) { return req->request_id() == id; });

      if (it != running_requests_.end()) {
        (*it)->finish();
        running_requests_.erase(it);
      }
    }
  }

  /**
   * @brief Get request by ID
   */
  InferenceRequestPtr get_request(i64 request_id) {
    auto it = request_map_.find(request_id);
    return (it != request_map_.end()) ? it->second : nullptr;
  }

  /**
   * @brief Check if there is work to do
   */
  bool has_work() const {
    return !running_requests_.empty() || !waiting_queue_.empty();
  }

  /**
   * @brief Get number of active sequences (RUNNING + WAITING)
   */
  i32 num_active_sequences() const {
    return static_cast<i32>(running_requests_.size() + waiting_queue_.size());
  }

  /**
   * @brief Get number of running requests
   */
  i32 num_running() const {
    return static_cast<i32>(running_requests_.size());
  }

  /**
   * @brief Get number of waiting requests
   */
  i32 num_waiting() const {
    return static_cast<i32>(waiting_queue_.size());
  }

  /**
   * @brief Get chunk size for prefill
   */
  i32 chunk_size() const {
    return chunk_size_;
  }

  /**
   * @brief Set chunk size for prefill
   */
  void set_chunk_size(i32 chunk_size) {
    chunk_size_ = chunk_size;
  }

  /**
   * @brief Get statistics
   */
  struct Stats {
    i32 num_running;
    i32 num_waiting;
    i32 num_finished;
    i32 total_requests;
  };

  Stats get_stats() const {
    Stats stats;
    stats.num_running = num_running();
    stats.num_waiting = num_waiting();
    stats.num_finished = 0;  // Count from request_map_
    for (const auto& [id, req] : request_map_) {
      if (req->is_finished()) {
        stats.num_finished++;
      }
    }
    stats.total_requests = static_cast<i32>(request_map_.size());
    return stats;
  }

  /**
   * @brief Clear all finished requests from memory
   */
  void clear_finished_requests() {
    auto it = request_map_.begin();
    while (it != request_map_.end()) {
      if (it->second->is_finished()) {
        it = request_map_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  i32 max_batch_size_;
  i32 max_sequences_;
  i32 chunk_size_;  // Prefill chunk size for chunked prefill optimization
  SchedulingPolicy policy_;
  i64 next_seq_id_;

  // Request queues
  std::deque<InferenceRequestPtr> waiting_queue_;
  std::vector<InferenceRequestPtr> running_requests_;

  // Request tracking
  std::unordered_map<i64, InferenceRequestPtr> request_map_;
};

}  // namespace scheduler
}  // namespace photon
