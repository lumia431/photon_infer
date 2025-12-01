/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file rmsnorm.hpp
 * @brief RMS normalization operator
 * @version 0.1.0
 */


#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// RMSNorm Operator
// ============================================================================

/**
 * @class RMSNormOp
 * @brief Root Mean Square Layer Normalization operator
 *
 * RMSNorm is a simplified variant of LayerNorm that normalizes inputs
 * using only the root mean square (RMS) statistic, without centering.
 * It's commonly used in LLaMA and other modern transformer models.
 *
 * Formula:
 * ```
 * rms = sqrt(mean(input^2) + eps)
 * output = weight * (input / rms)
 * ```
 *
 * Architecture:
 * - Input: [dim] or [batch × dim]
 * - Weight: [dim] (element-wise scaling factors)
 * - Output: [dim] or [batch × dim]
 *
 * Implementation strategies:
 * 1. Naive: Manual loop-based implementation
 * 2. Eigen: Vectorized array operations (SIMD)
 *
 * Example:
 * ```cpp
 * // Create RMSNorm with dim=512
 * RMSNormOp op(512);
 *
 * // Set weight (scaling factors)
 * auto weight = Tensor::create({512}, DataType::Float32).value();
 * // ... fill weight with values ...
 * op.set_weight(std::move(weight));
 * op.init();
 *
 * // Forward: normalize single vector [512]
 * auto input = Tensor::create({512}, DataType::Float32).value();
 * auto output = Tensor::create({512}, DataType::Float32).value();
 * op.forward(input, output);
 *
 * // Or batch: normalize [32 × 512] batch
 * auto batch_input = Tensor::create({32, 512}, DataType::Float32).value();
 * auto batch_output = Tensor::create({32, 512}, DataType::Float32).value();
 * op.forward(batch_input, batch_output);
 * ```
 */
class RMSNormOp : public ParameterizedOperator<RMSNormOp> {
 public:
  /**
   * @brief Construct RMSNorm operator
   *
   * @param dim Feature dimension to normalize
   * @param eps Small constant for numerical stability (default: 1e-5)
   * @param use_naive Use naive implementation (for benchmarking)
   */
  explicit RMSNormOp(i32 dim, f32 eps = 1e-5f, bool use_naive = false)
      : dim_(dim), eps_(eps), use_naive_(use_naive) {
    weights_.resize(1);
  }

  /**
   * @brief Set normalization weight (scaling factors)
   *
   * @param weight Tensor of shape [dim]
   * @return Result indicating success or error
   */
  Result<void> set_weight(Tensor weight) {
    // Validate weight shape
    if (weight.ndim() != 1) {
      return Err<void>(ErrorCode::InvalidShape,
                      "RMSNorm weight must be 1D tensor");
    }

    if (weight.dim(0) != dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Weight shape mismatch: expected [" + std::to_string(dim_) +
              "], got [" + std::to_string(weight.dim(0)) + "]");
    }

    if (weight.dtype() != DataType::Float32) {
      return Err<void>(ErrorCode::InvalidDtype,
                      "RMSNorm weight must be Float32");
    }

    // Auto-convert weight to operator's device if needed
    if (weight.device() != device_) {
      auto converted = weight.to(device_);
      if (!converted) {
        return Err<void>(converted.error());
      }
      weight = std::move(converted.value());
    }

    weights_[0] = std::move(weight);
    return Ok();
  }

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    if (!weights_initialized()) {
      return Err<void>(ErrorCode::InvalidOperator,
                      "RMSNorm weights not set");
    }
    return Ok();
  }

  /**
   * @brief Forward pass: normalize input using RMS
   *
   * Supports:
   * - Single vector: [dim] -> [dim]
   * - Batch: [batch × dim] -> [batch × dim]
   *
   * @param input Input tensor, shape [dim] or [batch × dim]
   * @param output Output tensor, same shape as input
   * @return Result indicating success or error
   */
  Result<void> forward(const Tensor& input, Tensor& output);

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "RMSNormOp";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Normalization;
  }

  /**
   * @brief Get normalization dimension
   */
  [[nodiscard]] i32 dim() const noexcept { return dim_; }

  /**
   * @brief Get epsilon value
   */
  [[nodiscard]] f32 eps() const noexcept { return eps_; }

  /**
   * @brief Check if using naive implementation
   */
  [[nodiscard]] bool is_naive() const noexcept { return use_naive_; }

 private:
  i32 dim_;         ///< Feature dimension
  f32 eps_;         ///< Epsilon for numerical stability
  bool use_naive_;  ///< Use naive implementation flag

  /**
   * @brief CPU forward implementation
   */
  Result<void> forward_cpu(const Tensor& input, Tensor& output);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief CUDA forward implementation
   */
  Result<void> forward_cuda(const Tensor& input, Tensor& output);
#endif
};

// Verify RMSNormOp satisfies Operator concept
static_assert(Operator<RMSNormOp>, "RMSNormOp must satisfy Operator concept");
static_assert(UnaryOperator<RMSNormOp>, "RMSNormOp must satisfy UnaryOperator concept");

}  // namespace photon

