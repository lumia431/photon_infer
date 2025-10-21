/**
 * @file swiglu.cpp
 * @brief SwiGLU activation operator implementation
 * @version 0.1.0
 */

#include "photon/ops/swiglu.hpp"
#include "photon/ops/kernels/swiglu_kernel.hpp"

namespace photon {

// ============================================================================
// SwiGLUOp Implementation
// ============================================================================

Result<void> SwiGLUOp::forward(const Tensor& input1, const Tensor& input2, Tensor& output) {
  // Check initialization
  if (!is_initialized()) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "SwiGLU operator not initialized");
  }

  // Validate input1 tensor
  if (input1.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input1 tensor is empty");
  }

  if (input1.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Input1 must be Float32");
  }

  if (input1.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input1 must be 1D [hidden_dim]");
  }

  if (input1.dim(0) != hidden_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Input1 dimension mismatch: expected " +
            std::to_string(hidden_dim_) + ", got " +
            std::to_string(input1.dim(0)));
  }

  // Validate input2 tensor
  if (input2.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input2 tensor is empty");
  }

  if (input2.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Input2 must be Float32");
  }

  if (input2.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input2 must be 1D [hidden_dim]");
  }

  if (input2.dim(0) != hidden_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Input2 dimension mismatch: expected " +
            std::to_string(hidden_dim_) + ", got " +
            std::to_string(input2.dim(0)));
  }

  // Validate output tensor
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  if (output.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Output must be Float32");
  }

  if (output.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Output must be 1D [hidden_dim]");
  }

  if (output.dim(0) != hidden_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Output dimension mismatch: expected " +
            std::to_string(hidden_dim_) + ", got " +
            std::to_string(output.dim(0)));
  }

  // Dispatch to device-specific implementation
  if (device_ == DeviceType::CPU && input1.device() == DeviceType::CPU &&
      input2.device() == DeviceType::CPU && output.device() == DeviceType::CPU) {
    return forward_cpu(input1, input2, output);
  }

#ifdef PHOTON_USE_CUDA
  if (device_ == DeviceType::CUDA && input1.device() == DeviceType::CUDA &&
      input2.device() == DeviceType::CUDA && output.device() == DeviceType::CUDA) {
    return forward_cuda(input1, input2, output);
  }
#endif

  return Err<void>(ErrorCode::DeviceMismatch,
                  "Input/output device mismatch with operator device");
}

Result<void> SwiGLUOp::forward_cpu(const Tensor& input1, const Tensor& input2, Tensor& output) {
  // Create spans for kernels
  std::span<const f32> input1_data(input1.ptr<f32>(), input1.size());
  std::span<const f32> input2_data(input2.ptr<f32>(), input2.size());
  std::span<f32> output_data(output.ptr<f32>(), output.size());

  // Dispatch based on implementation choice
  if (use_naive_) {
    kernels::swiglu_naive<f32>(
        input1_data, input2_data, output_data,
        hidden_dim_);
    return Ok();
  } else {
    return kernels::swiglu_eigen<f32>(
        input1_data, input2_data, output_data,
        hidden_dim_);
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> SwiGLUOp::forward_cuda(const Tensor& input1, const Tensor& input2, Tensor& output) {
  return Err<void>(ErrorCode::NotImplemented,
                  "CUDA swiglu not yet implemented");
}
#endif

}  // namespace photon

