/**
 * @file matmul.cpp
 * @brief Matrix multiplication operator implementation
 * @version 0.1.0
 */

#include "photon/ops/matmul.hpp"
#include "photon/ops/kernels/matmul_kernel.hpp"

namespace photon {

// ============================================================================
// MatMulOp Implementation
// ============================================================================

Result<void> MatMulOp::forward(const Tensor& input, Tensor& output) {
  // Check initialization
  if (!is_initialized()) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "MatMul operator not initialized");
  }

  // Validate input tensor
  if (input.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input tensor is empty");
  }

  if (input.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Input must be Float32");
  }

  // Input can be 1D [N] or 2D [B × N]
  if (input.ndim() != 1 && input.ndim() != 2) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input must be 1D [N] or 2D [B × N]");
  }

  // Check input dimension matches
  i32 input_last_dim = input.ndim() == 1 ? input.dim(0) : input.dim(1);
  if (input_last_dim != input_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Input last dimension mismatch: expected " +
            std::to_string(input_dim_) + ", got " +
            std::to_string(input_last_dim));
  }

  // Validate output tensor
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  if (output.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Output must be Float32");
  }

  // Output shape should match input batch dimension
  if (input.ndim() == 1) {
    // GEMV: [N] @ [M×N]^T -> [M]
    if (output.ndim() != 1) {
      return Err<void>(ErrorCode::InvalidShape,
                      "Output must be 1D for vector input");
    }
    if (output.dim(0) != output_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output dimension mismatch: expected " +
              std::to_string(output_dim_) + ", got " +
              std::to_string(output.dim(0)));
    }
  } else {
    // GEMM: [B×N] @ [M×N]^T -> [B×M]
    if (output.ndim() != 2) {
      return Err<void>(ErrorCode::InvalidShape,
                      "Output must be 2D for matrix input");
    }
    if (output.dim(0) != input.dim(0) || output.dim(1) != output_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output shape mismatch: expected [" +
              std::to_string(input.dim(0)) + " × " +
              std::to_string(output_dim_) + "], got [" +
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

Result<void> MatMulOp::forward_cpu(const Tensor& input, Tensor& output) {
  // Get weight tensor
  const Tensor& weight = weights_[0];

  // Create spans for kernels
  std::span<const f32> input_data(input.ptr<f32>(), input.size());
  std::span<const f32> weight_data(weight.ptr<f32>(), weight.size());
  std::span<f32> output_data(output.ptr<f32>(), output.size());

  // Dispatch based on input shape
  if (input.ndim() == 1) {
    // GEMV: [N] @ [M×N]^T -> [M]
    if (use_naive_) {
      kernels::matmul_gemv_naive<f32>(
          input_data, weight_data, output_data,
          output_dim_, input_dim_);
      return Ok();
    } else {
      return kernels::matmul_gemv_eigen<f32>(
          input_data, weight_data, output_data,
          output_dim_, input_dim_);
    }
  } else {
    // GEMM: [B×N] @ [M×N]^T -> [B×M]
    i32 batch_size = input.dim(0);
    if (use_naive_) {
      kernels::matmul_gemm_naive<f32>(
          input_data, weight_data, output_data,
          batch_size, output_dim_, input_dim_);
      return Ok();
    } else {
      return kernels::matmul_gemm_eigen<f32>(
          input_data, weight_data, output_data,
          batch_size, output_dim_, input_dim_);
    }
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> MatMulOp::forward_cuda(const Tensor& input, Tensor& output) {
  return Err<void>(ErrorCode::NotImplemented,
                  "CUDA matmul not yet implemented");
}
#endif

}  // namespace photon
