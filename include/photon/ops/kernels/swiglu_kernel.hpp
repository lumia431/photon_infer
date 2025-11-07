/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file swiglu_kernel.hpp
 * @brief CPU kernels for SwiGLU activation
 * @version 0.1.0
 */


#include <cmath>
#include <span>
#include <Eigen/Core>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels {

// ============================================================================
// Naive Implementation (Manual Loop)
// ============================================================================

/**
 * @brief Naive SwiGLU implementation with manual loops
 *
 * Computes SwiGLU activation element-wise:
 * ```
 * for i in 0..dim:
 *   silu = input1[i] / (1.0 + exp(-input1[i]))
 *   output[i] = silu * input2[i]
 * ```
 *
 * SiLU (Swish) activation:
 * - silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Complexity: O(dim)
 *
 * @param input1 Gate input [dim]
 * @param input2 Value input [dim]
 * @param output Output [dim]
 * @param dim Hidden dimension
 */
template <FloatingPoint T>
void swiglu_naive(
    std::span<const T> input1,
    std::span<const T> input2,
    std::span<T> output,
    i32 dim) noexcept {

  // Validate sizes
  if (input1.size() != static_cast<usize>(dim) ||
      input2.size() != static_cast<usize>(dim) ||
      output.size() != static_cast<usize>(dim)) {
    return;  // Silent failure in noexcept function
  }

  // Compute element-wise: output = SiLU(input1) * input2
  for (i32 i = 0; i < dim; ++i) {
    T x = input1[i];

    // SiLU(x) = x / (1 + exp(-x))
    T silu = x / (static_cast<T>(1.0) + std::exp(-x));

    // Gate the value input
    output[i] = silu * input2[i];
  }
}

// ============================================================================
// Optimized Implementation (Eigen)
// ============================================================================

/**
 * @brief Optimized SwiGLU using Eigen
 *
 * Uses Eigen's vectorized array operations for better performance:
 * - Vectorized exp() computation (SIMD)
 * - Element-wise operations without loops
 * - Cache-friendly memory access
 *
 * Performance: ~2-4x faster than naive for typical dimensions
 *
 * @param input1 Gate input [dim]
 * @param input2 Value input [dim]
 * @param output Output [dim]
 * @param dim Hidden dimension
 * @return Result indicating success or error
 */
template <FloatingPoint T>
Result<void> swiglu_eigen(
    std::span<const T> input1,
    std::span<const T> input2,
    std::span<T> output,
    i32 dim) noexcept {

  // Validate sizes
  if (input1.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input1 size mismatch in swiglu_eigen");
  }
  if (input2.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input2 size mismatch in swiglu_eigen");
  }
  if (output.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in swiglu_eigen");
  }

  // Create Eigen array maps (zero-copy views)
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> input1_arr(
      input1.data(), dim);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> input2_arr(
      input2.data(), dim);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> output_arr(
      output.data(), dim);

  // Compute SiLU(input1) = input1 / (1 + exp(-input1))
  // Using Eigen's array operations for vectorization
  auto silu = input1_arr / (static_cast<T>(1.0) + (-input1_arr).exp());

  // Gate with input2: output = SiLU(input1) * input2
  output_arr = silu * input2_arr;

  return Ok();
}

}  // namespace photon::kernels

