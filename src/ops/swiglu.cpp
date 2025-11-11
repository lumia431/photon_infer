/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file swiglu.cpp
 * @brief SwiGLU activation operator implementation
 * @version 0.1.0
 */

#include "photon/ops/swiglu.hpp"
#include "photon/ops/kernels/swiglu_kernel.hpp"

#ifdef PHOTON_USE_CUDA
#include "photon/ops/kernels/cuda/swiglu_kernel.cuh"
#endif

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

  // Support both 1D [hidden_dim] and 2D [batch, hidden_dim]
  if (input1.ndim() != 1 && input1.ndim() != 2) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input1 must be 1D [hidden_dim] or 2D [batch, hidden_dim]");
  }

  i32 input1_last_dim = input1.ndim() == 1 ? input1.dim(0) : input1.dim(1);
  if (input1_last_dim != hidden_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Input1 last dimension mismatch: expected " +
            std::to_string(hidden_dim_) + ", got " +
            std::to_string(input1_last_dim));
  }

  // Validate input2 tensor
  if (input2.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input2 tensor is empty");
  }

  if (input2.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Input2 must be Float32");
  }

  if (input2.ndim() != input1.ndim()) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Input2 must have same ndim as input1");
  }

  if (input2.ndim() == 1) {
    if (input2.dim(0) != hidden_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Input2 dimension mismatch: expected " +
              std::to_string(hidden_dim_) + ", got " +
              std::to_string(input2.dim(0)));
    }
  } else {
    if (input2.dim(0) != input1.dim(0) || input2.dim(1) != hidden_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Input2 shape mismatch");
    }
  }

  // Validate output tensor
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  if (output.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Output must be Float32");
  }

  if (output.ndim() != input1.ndim()) {
    return Err<void>(ErrorCode::InvalidShape,
                    "Output must have same ndim as inputs");
  }

  if (output.ndim() == 1) {
    if (output.dim(0) != hidden_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output dimension mismatch: expected " +
              std::to_string(hidden_dim_) + ", got " +
              std::to_string(output.dim(0)));
    }
  } else {
    if (output.dim(0) != input1.dim(0) || output.dim(1) != hidden_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Output shape mismatch");
    }
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

  if (input1.ndim() == 1) {
    // Single vector
    if (use_naive_) {
      kernels::swiglu_naive<f32>(
          input1_data, input2_data, output_data,
          hidden_dim_);
      return Ok();
    } else {
#ifdef PHOTON_USE_EIGEN
      return kernels::swiglu_eigen<f32>(
          input1_data, input2_data, output_data,
          hidden_dim_);
#else
      return Err<void>(ErrorCode::NotImplemented,
                      "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
    }
  } else {
    // Batched: process each row
    i32 batch_size = input1.dim(0);
    for (i32 b = 0; b < batch_size; ++b) {
      const f32* in1_row = input1.ptr<f32>() + b * hidden_dim_;
      const f32* in2_row = input2.ptr<f32>() + b * hidden_dim_;
      f32* out_row = output.ptr<f32>() + b * hidden_dim_;

      std::span<const f32> in1_span(in1_row, hidden_dim_);
      std::span<const f32> in2_span(in2_row, hidden_dim_);
      std::span<f32> out_span(out_row, hidden_dim_);

      if (use_naive_) {
        kernels::swiglu_naive<f32>(
            in1_span, in2_span, out_span,
            hidden_dim_);
      } else {
#ifdef PHOTON_USE_EIGEN
        auto result = kernels::swiglu_eigen<f32>(
            in1_span, in2_span, out_span,
            hidden_dim_);
        if (!result) return result;
#else
        return Err<void>(ErrorCode::NotImplemented,
                        "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
      }
    }
    return Ok();
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> SwiGLUOp::forward_cuda(const Tensor& input1, const Tensor& input2, Tensor& output) {
  if (input1.ndim() == 1) {
    // Single vector
    std::span<const f32> input1_data(input1.ptr<f32>(), input1.size());
    std::span<const f32> input2_data(input2.ptr<f32>(), input2.size());
    std::span<f32> output_data(output.ptr<f32>(), output.size());

    return kernels::cuda::swiglu_cuda_launch(
        input1_data, input2_data, output_data,
        hidden_dim_, nullptr);
  } else {
    // Batched: use true batched kernel
    i32 batch_size = input1.dim(0);
    return kernels::cuda::swiglu_batched_cuda_launch(
        input1.ptr<f32>(),
        input2.ptr<f32>(),
        output.ptr<f32>(),
        batch_size,
        hidden_dim_,
        nullptr);
  }
}
#endif

}  // namespace photon

