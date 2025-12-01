/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file quant.cpp
 * @brief Implementation of quantization utilities
 * @version 0.1.0
 */

#include "photon/core/quant.hpp"
#include <cmath>
#include <numeric>
#include <sstream>

namespace photon {

// ============================================================================
// Group-wise Quantization Implementation
// ============================================================================

Result<std::pair<std::vector<i8>, QuantParams>>
quantize_weights(std::span<const f32> weights, i32 group_size) {
  // Validate input
  if (weights.empty()) {
    return Err<std::pair<std::vector<i8>, QuantParams>>(
        ErrorCode::InvalidArgument, "Input weights are empty");
  }

  if (group_size <= 0) {
    return Err<std::pair<std::vector<i8>, QuantParams>>(
        ErrorCode::InvalidArgument,
        "Group size must be positive, got " + std::to_string(group_size));
  }

  const usize total_size = weights.size();
  const i32 num_groups = (total_size + group_size - 1) / group_size;

  // Allocate output buffers
  std::vector<i8> quantized(total_size);
  std::vector<f32> scales(num_groups);

  // Quantize each group
  for (i32 g = 0; g < num_groups; ++g) {
    const usize group_start = g * group_size;
    const usize group_end = std::min(group_start + group_size, total_size);
    const usize current_group_size = group_end - group_start;

    // Get group data
    auto group_data = weights.subspan(group_start, current_group_size);

    // Compute scale for this group
    f32 max_abs = compute_absmax(group_data);
    f32 scale = compute_scale(max_abs);
    scales[g] = scale;

    // Quantize each element in the group
    for (usize i = 0; i < current_group_size; ++i) {
      quantized[group_start + i] = quantize_value(group_data[i], scale);
    }
  }

  // Create QuantParams
  QuantParams params;
  params.group_size = group_size;
  params.num_groups = num_groups;
  params.scales = std::move(scales);

  return Ok(std::make_pair(std::move(quantized), std::move(params)));
}

Result<std::vector<f32>>
dequantize_weights(std::span<const i8> quantized, const QuantParams& params) {
  // Validate params
  if (!params.is_valid()) {
    return Err<std::vector<f32>>(
        ErrorCode::InvalidArgument, "Invalid quantization parameters");
  }

  const usize total_size = quantized.size();
  std::vector<f32> dequantized(total_size);

  // Dequantize each group
  for (i32 g = 0; g < params.num_groups; ++g) {
    const usize group_start = g * params.group_size;
    const usize group_end = std::min(group_start + params.group_size, total_size);
    const f32 scale = params.scales[g];

    for (usize i = group_start; i < group_end; ++i) {
      dequantized[i] = dequantize_value(quantized[i], scale);
    }
  }

  return Ok(std::move(dequantized));
}

f32 compute_quantization_error(
    std::span<const f32> original,
    std::span<const f32> dequantized) noexcept {

  if (original.size() != dequantized.size() || original.empty()) {
    return 0.0f;
  }

  // Compute RMSE
  f32 sum_squared_error = 0.0f;
  for (usize i = 0; i < original.size(); ++i) {
    f32 error = original[i] - dequantized[i];
    sum_squared_error += error * error;
  }

  return std::sqrt(sum_squared_error / static_cast<f32>(original.size()));
}

Result<std::pair<Tensor, QuantParams>>
quantize_tensor(const Tensor& tensor, i32 group_size) {
  // Validate tensor
  if (tensor.empty()) {
    return Err<std::pair<Tensor, QuantParams>>(
        ErrorCode::InvalidArgument, "Input tensor is empty");
  }

  if (tensor.dtype() != DataType::Float32) {
    return Err<std::pair<Tensor, QuantParams>>(
        ErrorCode::InvalidDtype, "Input tensor must be Float32");
  }

  // Get tensor data
  std::span<const f32> weights(tensor.ptr<f32>(), tensor.size());

  // Quantize
  auto result = quantize_weights(weights, group_size);
  if (!result) {
    return Err<std::pair<Tensor, QuantParams>>(result.error());
  }

  auto [quantized_data, params] = std::move(result.value());

  // Create quantized tensor
  auto quant_tensor_result = Tensor::create(
      std::vector<i32>(tensor.dims().begin(), tensor.dims().end()),
      DataType::Int8,
      tensor.device(),
      true);

  if (!quant_tensor_result) {
    return Err<std::pair<Tensor, QuantParams>>(quant_tensor_result.error());
  }

  Tensor quant_tensor = std::move(quant_tensor_result.value());

  // Copy quantized data
  if (tensor.device() == DeviceType::CPU) {
    std::memcpy(quant_tensor.ptr<i8>(), quantized_data.data(),
                quantized_data.size() * sizeof(i8));
  } else {
    // For CUDA, need to copy via cudaMemcpy
    // TODO: Implement CUDA copy in separate PR
    return Err<std::pair<Tensor, QuantParams>>(
        ErrorCode::NotImplemented, "CUDA quantization not yet implemented");
  }

  return Ok(std::make_pair(std::move(quant_tensor), std::move(params)));
}

// ============================================================================
// Quantization Statistics
// ============================================================================

std::string QuantStats::to_string() const {
  std::ostringstream oss;
  oss << "QuantStats {\n"
      << "  RMSE: " << rmse << "\n"
      << "  Max Error: " << max_error << "\n"
      << "  Compression Ratio: " << compression_ratio << "x\n"
      << "  Num Groups: " << num_groups << "\n"
      << "}";
  return oss.str();
}

QuantStats compute_quant_stats(
    std::span<const f32> original,
    std::span<const i8> quantized,
    const QuantParams& params) noexcept {

  QuantStats stats;
  stats.num_groups = params.num_groups;

  if (original.empty() || quantized.empty()) {
    return stats;
  }

  // Compute compression ratio
  // float32 = 4 bytes, int8 = 1 byte, scales = num_groups * 4 bytes
  usize original_bytes = original.size() * sizeof(f32);
  usize quantized_bytes = quantized.size() * sizeof(i8) +
                          params.scales.size() * sizeof(f32);
  stats.compression_ratio = static_cast<f32>(original_bytes) /
                           static_cast<f32>(quantized_bytes);

  // Dequantize for error computation
  auto dequant_result = dequantize_weights(quantized, params);
  if (!dequant_result) {
    return stats;
  }

  auto dequantized = std::move(dequant_result.value());

  // Compute RMSE
  stats.rmse = compute_quantization_error(original, dequantized);

  // Compute max error
  stats.max_error = 0.0f;
  for (usize i = 0; i < original.size(); ++i) {
    f32 error = std::abs(original[i] - dequantized[i]);
    stats.max_error = std::max(stats.max_error, error);
  }

  return stats;
}

}  // namespace photon
