/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file swiglu.hpp
 * @brief SwiGLU activation operator
 * @version 0.1.0
 */


#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// SwiGLU Operator
// ============================================================================

/**
 * @class SwiGLUOp
 * @brief SwiGLU (Swish-Gated Linear Unit) activation function
 *
 * SwiGLU is a gated activation function used in LLaMA's feed-forward network.
 * It combines the Swish/SiLU activation with gating mechanism.
 *
 * Formula:
 * ```
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * output = SiLU(input1) * input2
 * ```
 *
 * This is a variant of GLU (Gated Linear Unit) that uses SiLU instead of
 * sigmoid as the activation function.
 *
 * Architecture:
 * - Input1: [hidden_dim] - Gate input (will be activated)
 * - Input2: [hidden_dim] - Value input (will be gated)
 * - Output: [hidden_dim] - Gated output
 *
 * Typical usage in LLaMA FFN:
 * ```
 * x1 = MatMul(x, W_gate)     # [hidden_dim]
 * x2 = MatMul(x, W_up)       # [hidden_dim]
 * output = SwiGLU(x1, x2)    # [hidden_dim]
 * ```
 *
 * Example:
 * ```cpp
 * // LLaMA 3.2-1B: hidden_dim = 8192
 * SwiGLUOp swiglu(8192);
 * swiglu.init();
 *
 * auto input1 = Tensor::create({8192}, DataType::Float32).value();
 * auto input2 = Tensor::create({8192}, DataType::Float32).value();
 * auto output = Tensor::create({8192}, DataType::Float32).value();
 *
 * swiglu.forward(input1, input2, output);
 * ```
 */
class SwiGLUOp : public OperatorBase<SwiGLUOp> {
 public:
  /**
   * @brief Construct SwiGLU operator
   *
   * @param hidden_dim Hidden dimension of the inputs
   * @param use_naive Use naive implementation (for benchmarking)
   */
  explicit SwiGLUOp(i32 hidden_dim, bool use_naive = false)
      : hidden_dim_(hidden_dim), use_naive_(use_naive) {}

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    // No special initialization needed for SwiGLU
    return Ok();
  }

  /**
   * @brief Forward pass: output = SiLU(input1) * input2
   *
   * Applies SwiGLU activation element-wise:
   * - Compute SiLU(input1) = input1 / (1 + exp(-input1))
   * - Multiply by input2
   *
   * @param input1 Gate input tensor [hidden_dim]
   * @param input2 Value input tensor [hidden_dim]
   * @param output Output tensor [hidden_dim]
   * @return Result indicating success or error
   */
  Result<void> forward(const Tensor& input1, const Tensor& input2, Tensor& output);

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "SwiGLUOp";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Activation;
  }

  /**
   * @brief Get hidden dimension
   */
  [[nodiscard]] i32 hidden_dim() const noexcept { return hidden_dim_; }

  /**
   * @brief Check if using naive implementation
   */
  [[nodiscard]] bool is_naive() const noexcept { return use_naive_; }

 private:
  i32 hidden_dim_;   ///< Hidden dimension
  bool use_naive_;   ///< Use naive implementation flag

  /**
   * @brief CPU forward implementation
   */
  Result<void> forward_cpu(const Tensor& input1, const Tensor& input2, Tensor& output);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief CUDA forward implementation
   */
  Result<void> forward_cuda(const Tensor& input1, const Tensor& input2, Tensor& output);
#endif
};

// Verify SwiGLUOp satisfies Operator concept
static_assert(Operator<SwiGLUOp>, "SwiGLUOp must satisfy Operator concept");
static_assert(BinaryOperator<SwiGLUOp>, "SwiGLUOp must satisfy BinaryOperator concept");

}  // namespace photon

