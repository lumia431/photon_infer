/**
 * @file rmsnorm.cpp
 * @brief RMS normalization operator implementation
 * @version 0.1.0
 */

#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/kernels/rmsnorm_kernel.hpp"

namespace photon {

// ============================================================================
// RMSNormOp Implementation
// ============================================================================

Result<void> RMSNormOp::forward(const Tensor& input, Tensor& output) {
  // Check initialization
  if (!is_initialized()) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "RMSNorm operator not initialized");
  }

  // Validate input tensor
  if (input.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input tensor is empty");
  }

  if (input.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Input must be Float32");
  }

  // Input can be 1D [dim] or 2D [batch × dim]
  if (input.ndim() != 1 && input.ndim() != 2) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input must be 1D [dim] or 2D [batch × dim]");
  }

  // Check dimension matches
  i32 input_last_dim = input.ndim() == 1 ? input.dim(0) : input.dim(1);
  if (input_last_dim != dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Input last dimension mismatch: expected " +
            std::to_string(dim_) + ", got " +
            std::to_string(input_last_dim));
  }

  // Validate output tensor
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  if (output.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Output must be Float32");
  }

  // Output shape must match input shape
  if (output.ndim() != input.ndim()) {
    return Err<void>(ErrorCode::ShapeMismatch,
                    "Output shape must match input shape");
  }

  if (input.ndim() == 1) {
    // Single vector: [dim]
    if (output.dim(0) != dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output dimension mismatch: expected " +
              std::to_string(dim_) + ", got " +
              std::to_string(output.dim(0)));
    }
  } else {
    // Batch: [batch × dim]
    if (output.dim(0) != input.dim(0) || output.dim(1) != dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output shape mismatch: expected [" +
              std::to_string(input.dim(0)) + " × " +
              std::to_string(dim_) + "], got [" +
              std::to_string(output.dim(0)) + " × " +
              std::to_string(output.dim(1)) + "]");
    }
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

Result<void> RMSNormOp::forward_cpu(const Tensor& input, Tensor& output) {
  // Get weight tensor
  const Tensor& weight = weights_[0];

  // Create spans for kernels
  std::span<const f32> input_data(input.ptr<f32>(), input.size());
  std::span<const f32> weight_data(weight.ptr<f32>(), weight.size());
  std::span<f32> output_data(output.ptr<f32>(), output.size());

  // Dispatch based on input shape
  if (input.ndim() == 1) {
    // Single vector normalization: [dim] -> [dim]
    if (use_naive_) {
      kernels::rmsnorm_naive<f32>(
          input_data, weight_data, output_data,
          dim_, eps_);
      return Ok();
    } else {
      return kernels::rmsnorm_eigen<f32>(
          input_data, weight_data, output_data,
          dim_, eps_);
    }
  } else {
    // Batch normalization: [batch × dim] -> [batch × dim]
    i32 batch_size = input.dim(0);
    if (use_naive_) {
      kernels::rmsnorm_batch_naive<f32>(
          input_data, weight_data, output_data,
          batch_size, dim_, eps_);
      return Ok();
    } else {
      return kernels::rmsnorm_batch_eigen<f32>(
          input_data, weight_data, output_data,
          batch_size, dim_, eps_);
    }
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> RMSNormOp::forward_cuda(const Tensor& input, Tensor& output) {
  return Err<void>(ErrorCode::NotImplemented,
                  "CUDA rmsnorm not yet implemented");
}
#endif

}  // namespace photon
