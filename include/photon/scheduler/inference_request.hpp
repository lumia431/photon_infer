/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file inference_request.hpp
 * @brief Inference request object for continuous batching
 *
 * Inspired by vLLM's request.py, this implements request state management
 * for dynamic batch scheduling.
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>

#include "photon/core/types.hpp"

namespace photon {
namespace scheduler {

/**
 * @brief Request state machine
 *
 * State transitions:
 *   WAITING -> RUNNING -> FINISHED
 *   RUNNING -> PREEMPTED -> WAITING -> RUNNING
 */
enum class RequestState {
  WAITING,    // In queue, not yet scheduled
  RUNNING,    // Currently being processed
  FINISHED,   // Generation complete
  PREEMPTED,  // Temporarily paused (for resource management)
};

/**
 * @brief Inference request for continuous batching
 *
 * Tracks all state needed for token-level scheduling:
 * - num_computed_tokens: How many tokens have been computed (key for resume)
 * - arrival_time: When request arrived (for latency tracking)
 * - Generated tokens: Partial results
 */
class InferenceRequest {
 public:
  /**
   * @brief Create a new inference request
   *
   * @param request_id Unique request identifier
   * @param prompt Input prompt string
   * @param prompt_tokens Tokenized prompt
   * @param max_new_tokens Maximum tokens to generate
   */
  InferenceRequest(
      i64 request_id,
      std::string prompt,
      std::vector<i32> prompt_tokens,
      i32 max_new_tokens)
      : request_id_(request_id),
        prompt_(std::move(prompt)),
        prompt_tokens_(std::move(prompt_tokens)),
        max_new_tokens_(max_new_tokens),
        state_(RequestState::WAITING),
        num_computed_tokens_(0),
        arrival_time_(std::chrono::high_resolution_clock::now()) {}

  // Getters
  i64 request_id() const { return request_id_; }
  const std::string& prompt() const { return prompt_; }
  const std::vector<i32>& prompt_tokens() const { return prompt_tokens_; }
  i32 max_new_tokens() const { return max_new_tokens_; }
  RequestState state() const { return state_; }
  i32 num_computed_tokens() const { return num_computed_tokens_; }
  const std::vector<i32>& generated_tokens() const { return generated_tokens_; }

  /**
   * @brief Get total tokens (prompt + generated)
   */
  i32 total_tokens() const {
    return static_cast<i32>(prompt_tokens_.size()) + static_cast<i32>(generated_tokens_.size());
  }

  /**
   * @brief Get prompt length
   */
  i32 prompt_len() const {
    return static_cast<i32>(prompt_tokens_.size());
  }

  /**
   * @brief Get number of generated tokens
   */
  i32 num_generated() const {
    return static_cast<i32>(generated_tokens_.size());
  }

  /**
   * @brief Check if request is finished
   */
  bool is_finished() const {
    return state_ == RequestState::FINISHED;
  }

  /**
   * @brief Check if in prefill phase (no tokens generated yet)
   */
  bool is_prefill() const {
    return num_computed_tokens_ < prompt_len();
  }

  /**
   * @brief Check if in decode phase (generating new tokens)
   */
  bool is_decode() const {
    return num_computed_tokens_ >= prompt_len() && !is_finished();
  }

  /**
   * @brief Get number of tokens remaining to compute in prefill
   */
  i32 prefill_remaining() const {
    return std::max(0, prompt_len() - num_computed_tokens_);
  }

  /**
   * @brief Get arrival time
   */
  auto arrival_time() const { return arrival_time_; }

  /**
   * @brief Get latency since arrival (seconds)
   */
  double latency_seconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - arrival_time_).count();
  }

  // State manipulation
  void set_state(RequestState state) { state_ = state; }

  /**
   * @brief Mark request as running
   */
  void start_running() {
    state_ = RequestState::RUNNING;
    if (start_time_.time_since_epoch().count() == 0) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
  }

  /**
   * @brief Mark request as finished
   */
  void finish() {
    state_ = RequestState::FINISHED;
    finish_time_ = std::chrono::high_resolution_clock::now();
  }

  /**
   * @brief Add a newly generated token
   *
   * @param token Token ID to add
   * @param eos_token_id EOS token ID for termination check
   * @return true if generation should continue, false if finished
   */
  bool add_token(i32 token, i32 eos_token_id) {
    generated_tokens_.push_back(token);
    num_computed_tokens_++;

    // Check termination conditions
    if (token == eos_token_id || num_generated() >= max_new_tokens_) {
      finish();
      return false;
    }
    return true;
  }

  /**
   * @brief Increment computed tokens (for prefill)
   *
   * @param count Number of tokens computed
   */
  void add_computed_tokens(i32 count) {
    num_computed_tokens_ += count;
  }

  /**
   * @brief Get current position for next forward pass
   *
   * This is critical for KV cache indexing.
   */
  i32 current_position() const {
    return num_computed_tokens_;
  }

  /**
   * @brief Get next token to process
   *
   * During prefill: returns prompt_tokens[num_computed_tokens]
   * During decode: returns last generated token
   */
  i32 next_token() const {
    if (is_prefill()) {
      return prompt_tokens_[num_computed_tokens_];
    } else {
      // During decode, use last generated token
      if (!generated_tokens_.empty()) {
        return generated_tokens_.back();
      } else {
        // First decode step: use last prompt token
        return prompt_tokens_.back();
      }
    }
  }

  /**
   * @brief Preempt this request (pause execution)
   */
  void preempt() {
    state_ = RequestState::PREEMPTED;
  }

  /**
   * @brief Resume from preemption
   */
  void resume() {
    state_ = RequestState::WAITING;
  }

  /**
   * @brief Get time spent in execution (excluding waiting)
   */
  double execution_time_seconds() const {
    if (start_time_.time_since_epoch().count() == 0) return 0.0;
    auto end = (finish_time_.time_since_epoch().count() > 0)
                ? finish_time_
                : std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start_time_).count();
  }

 private:
  i64 request_id_;
  std::string prompt_;
  std::vector<i32> prompt_tokens_;
  i32 max_new_tokens_;

  RequestState state_;
  i32 num_computed_tokens_;  // Critical: tracks resume point
  std::vector<i32> generated_tokens_;

  std::chrono::high_resolution_clock::time_point arrival_time_;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point finish_time_;
};

using InferenceRequestPtr = std::shared_ptr<InferenceRequest>;

}  // namespace scheduler
}  // namespace photon
