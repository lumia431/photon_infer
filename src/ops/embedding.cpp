/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file embedding.cpp
 * @brief Embedding operator implementation
 * @version 0.1.0
 */

#include "photon/ops/embedding.hpp"
#include "photon/ops/kernels/embedding_kernel.hpp"

#ifdef PHOTON_USE_CUDA
#include "photon/ops/kernels/cuda/embedding_kernel.cuh"
#endif

namespace photon {

// ============================================================================
// EmbeddingOp Implementation
// ============================================================================

Result<void> EmbeddingOp::forward(const Tensor& input, Tensor& output) {
  // Check initialization
  if (!is_initialized()) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "Embedding operator not initialized");
  }

  // Validate input tensor
  if (input.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input tensor is empty");
  }

  if (input.dtype() != DataType::Int32) {
    return Err<void>(ErrorCode::InvalidDtype,
                    "Input must be Int32 (token IDs)");
  }

  if (input.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input must be 1D tensor [batch_size] or [seq_len]");
  }

  // Validate output tensor
  const usize batch_size = input.size();
  const usize expected_output_size = batch_size * static_cast<usize>(embedding_dim_);

  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  if (output.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Output must be Float32");
  }

  if (output.ndim() != 2) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Output must be 2D tensor [batch_size × embedding_dim]");
  }

  if (output.dim(0) != static_cast<i32>(batch_size) ||
      output.dim(1) != embedding_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Output shape mismatch: expected [" + std::to_string(batch_size) +
            " × " + std::to_string(embedding_dim_) + "], got [" +
            std::to_string(output.dim(0)) + " × " +
            std::to_string(output.dim(1)) + "]");
  }

  // Dispatch to device-specific implementation
  if (device_ == DeviceType::CPU && input.device() == DeviceType::CPU &&
      output.device() == DeviceType::CPU) {
    return forward_cpu(input, output);
  }

#ifdef PHOTON_USE_CUDA
  if (device_ == DeviceType::CUDA && input.device() == DeviceType::CUDA &&
      output.device() == DeviceType::CUDA) {
    return forward_cuda(input, output);
  }
#endif

  return Err<void>(ErrorCode::DeviceMismatch,
                  "Input/output device mismatch with operator device");
}

Result<void> EmbeddingOp::forward_cpu(const Tensor& input, Tensor& output) {
  // Get weight tensor
  const Tensor& weight = weights_[0];

  // Create spans for kernel
  std::span<const i32> tokens(input.ptr<i32>(), input.size());
  std::span<const f32> weight_data(weight.ptr<f32>(), weight.size());
  std::span<f32> output_data(output.ptr<f32>(), output.size());

  // Call CPU kernel
#ifdef PHOTON_USE_EIGEN
  return kernels::embedding_forward_cpu<i32, f32>(
      tokens, weight_data, output_data, vocab_size_, embedding_dim_);
#else
  // Use unchecked version when Eigen is disabled (assumes valid indices)
  kernels::embedding_forward_cpu_unchecked<i32, f32>(
      tokens, weight_data, output_data, embedding_dim_);
  return Ok();
#endif
}

#ifdef PHOTON_USE_CUDA
Result<void> EmbeddingOp::forward_cuda(const Tensor& input, Tensor& output) {
  // Get weight tensor
  const Tensor& weight = weights_[0];

  // Create spans for CUDA kernel launch
  std::span<const i32> tokens(input.ptr<i32>(), input.size());
  std::span<const f32> weight_data(weight.ptr<f32>(), weight.size());
  std::span<f32> output_data(output.ptr<f32>(), output.size());

  // Launch CUDA kernel (using standard approach)
  i32 num_tokens = static_cast<i32>(input.size());
  return kernels::cuda::embedding_cuda_launch(
      tokens, weight_data, output_data,
      num_tokens, vocab_size_, embedding_dim_,
      nullptr);  // stream = nullptr for now
}
#endif

}  // namespace photon
