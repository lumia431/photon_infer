/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file quant.hpp
 * @brief Quantization utilities for int8 inference
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-10-25
 *
 * This file provides quantization utilities for converting float32 weights
 * to int8 quantized representation using group-wise symmetric quantization.
 *
 * Design Philosophy:
 * - Group-wise quantization for better accuracy
 * - Symmetric quantization (no zero-point) for simplicity
 * - Dynamic dequantization during inference
 * - Modern C++20 features: concepts, ranges, constexpr
 */


#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <span>
#include <vector>

#include "error.hpp"
#include "tensor.hpp"
#include "types.hpp"

namespace photon {

// ============================================================================
// Quantization Parameters
// ============================================================================

/**
 * @struct QuantParams
 * @brief Quantization parameters for a weight tensor
 *
 * For group-wise symmetric quantization:
 * - Each group has its own scale factor
 * - No zero-point (symmetric around 0)
 * - Formula: dequant_value = scale * int8_value
 *
 * Example:
 * ```cpp
 * QuantParams params;
 * params.group_size = 128;
 * params.num_groups = weight_size / 128;
 * params.scales = compute_scales(weights, 128);
 * ```
 */
struct QuantParams {
  i32 group_size = 128;              ///< Number of elements per group
  i32 num_groups = 0;                ///< Total number of groups
  std::vector<f32> scales;           ///< Scale factors [num_groups]

  /**
   * @brief Check if parameters are valid
   */
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return group_size > 0 &&
           num_groups > 0 &&
           scales.size() == static_cast<usize>(num_groups);
  }

  /**
   * @brief Get total number of quantized elements
   */
  [[nodiscard]] constexpr i32 total_size() const noexcept {
    return group_size * num_groups;
  }
};

// ============================================================================
// Quantization Utilities
// ============================================================================

/**
 * @brief Compute the maximum absolute value in a range
 *
 * Modern C++20 implementation using ranges and concepts.
 *
 * @param data Input data range
 * @return Maximum absolute value
 */
template <std::ranges::range R>
  requires std::floating_point<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto compute_absmax(R&& data) noexcept {
  using T = std::ranges::range_value_t<R>;

  if (std::ranges::empty(data)) {
    return T{0};
  }

  // Use ranges algorithm for elegance
  auto abs_view = data | std::views::transform([](T x) {
    return std::abs(x);
  });

  return std::ranges::max(abs_view);
}

/**
 * @brief Compute scale factor for symmetric quantization
 *
 * For int8: range is [-127, 127] (avoid -128 for symmetry)
 * scale = max_abs_value / 127.0
 *
 * @param max_abs Maximum absolute value in the group
 * @return Scale factor
 */
[[nodiscard]] constexpr f32 compute_scale(f32 max_abs) noexcept {
  constexpr f32 INT8_RANGE = 127.0f;

  if (max_abs < 1e-8f) {
    // Avoid division by zero for all-zero groups
    return 1.0f;
  }

  return max_abs / INT8_RANGE;
}

/**
 * @brief Quantize a single float value to int8
 *
 * @param value Float value to quantize
 * @param scale Scale factor
 * @return Quantized int8 value, clamped to [-127, 127]
 */
[[nodiscard]] inline i8 quantize_value(f32 value, f32 scale) noexcept {
  if (scale < 1e-8f) {
    return 0;
  }

  // Quantize and clamp to [-127, 127]
  f32 quantized = std::round(value / scale);
  quantized = std::clamp(quantized, -127.0f, 127.0f);

  return static_cast<i8>(quantized);
}

/**
 * @brief Dequantize a single int8 value to float
 *
 * @param value Quantized int8 value
 * @param scale Scale factor
 * @return Dequantized float value
 */
[[nodiscard]] constexpr f32 dequantize_value(i8 value, f32 scale) noexcept {
  return static_cast<f32>(value) * scale;
}

// ============================================================================
// Group-wise Quantization
// ============================================================================

/**
 * @brief Perform group-wise symmetric quantization on weight tensor
 *
 * This function:
 * 1. Divides weights into groups of size `group_size`
 * 2. Computes scale factor for each group
 * 3. Quantizes each element to int8
 *
 * Algorithm (Modern C++20 style):
 * ```cpp
 * for each group in weights:
 *   max_abs = max(abs(group))
 *   scale = max_abs / 127.0
 *   for each element in group:
 *     quantized[i] = clamp(round(element / scale), -127, 127)
 * ```
 *
 * @param weights Input float weights [N]
 * @param group_size Number of elements per group
 * @return Result containing {quantized weights, QuantParams}
 */
[[nodiscard]] Result<std::pair<std::vector<i8>, QuantParams>>
quantize_weights(
    std::span<const f32> weights,
    i32 group_size = 128);

/**
 * @brief Dequantize int8 weights back to float32 (for validation)
 *
 * @param quantized Quantized int8 weights
 * @param params Quantization parameters
 * @return Dequantized float32 weights
 */
[[nodiscard]] Result<std::vector<f32>>
dequantize_weights(
    std::span<const i8> quantized,
    const QuantParams& params);

/**
 * @brief Compute quantization error (RMSE)
 *
 * Useful for validating quantization quality.
 *
 * @param original Original float32 weights
 * @param dequantized Dequantized float32 weights
 * @return Root Mean Square Error
 */
[[nodiscard]] f32 compute_quantization_error(
    std::span<const f32> original,
    std::span<const f32> dequantized) noexcept;

/**
 * @brief Convert a float32 tensor to quantized int8 tensor
 *
 * High-level API for quantizing entire tensors.
 *
 * @param tensor Input float32 tensor
 * @param group_size Group size for quantization
 * @return Result containing {quantized tensor, QuantParams}
 */
[[nodiscard]] Result<std::pair<Tensor, QuantParams>>
quantize_tensor(
    const Tensor& tensor,
    i32 group_size = 128);

// ============================================================================
// Quantization Statistics
// ============================================================================

/**
 * @struct QuantStats
 * @brief Statistics about quantization quality
 */
struct QuantStats {
  f32 rmse = 0.0f;              ///< Root Mean Square Error
  f32 max_error = 0.0f;         ///< Maximum absolute error
  f32 compression_ratio = 0.0f; ///< Memory compression ratio
  i32 num_groups = 0;           ///< Number of groups

  /**
   * @brief Convert stats to human-readable string
   */
  [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Compute detailed quantization statistics
 *
 * @param original Original float32 weights
 * @param quantized Quantized int8 weights
 * @param params Quantization parameters
 * @return Quantization statistics
 */
[[nodiscard]] QuantStats compute_quant_stats(
    std::span<const f32> original,
    std::span<const i8> quantized,
    const QuantParams& params) noexcept;

}  // namespace photon

