/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file matmul.cpp
 * @brief Matrix multiplication operator implementation
 * @version 0.1.0
 */

#include "photon/ops/matmul.hpp"
#include "photon/ops/kernels/matmul_kernel.hpp"

#ifdef PHOTON_USE_CUDA
#include "photon/ops/kernels/cuda/matmul_gemv_quant.cuh"
#include "photon/ops/kernels/cuda/matmul_dequant_cached.cuh"
#include <cublas_v2.h>
#endif

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
#ifdef PHOTON_USE_EIGEN
      return kernels::matmul_gemv_eigen<f32>(
          input_data, weight_data, output_data,
          output_dim_, input_dim_);
#else
      return Err<void>(ErrorCode::NotImplemented,
                      "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
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
#ifdef PHOTON_USE_EIGEN
      return kernels::matmul_gemm_eigen<f32>(
          input_data, weight_data, output_data,
          batch_size, output_dim_, input_dim_);
#else
      return Err<void>(ErrorCode::NotImplemented,
                      "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
    }
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> MatMulOp::forward_cuda(const Tensor& input, Tensor& output) {
  // Only quantized path is supported
  if (!is_quantized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "FP32 path is not supported. Please quantize the model weights first by calling quantize_weight().");
  }

  // Dispatch based on input shape
  if (input.ndim() == 1) {
    // GEMV: [N] @ [M×N]^T -> [M]
    i32 M = input_dim_;   // Input dimension
    i32 K = output_dim_;  // Output dimension (number of rows in weight)

    // Quantized path: int8 weights with dynamic dequantization
    const Tensor& weight = weights_[0];

    return kernels::cuda::matmul_gemv_quant_launch(
        input.ptr<f32>(), input.size(),
        weight.ptr<i8>(), weight.size(),
        scale_tensor_.ptr<f32>(), scale_tensor_.size(),
        quant_params_.group_size,
        output.ptr<f32>(), output.size(),
        M, K,
        nullptr);  // stream = nullptr for now
  } else {
    // GEMM: [B×N] @ [M×N]^T -> [B×M]
    // Use cuBLAS for optimal batched matrix multiplication
    i32 batch_size = input.dim(0);
    i32 M = input_dim_;   // Input dimension
    i32 K = output_dim_;  // Output dimension

    // Quantized path: use cached dequantization + cuBLAS FP32 GEMM
    const Tensor& weight = weights_[0];

    // Use dequant-cached kernel (cuBLAS handle is always set in production)
    return kernels::cuda::matmul_gemm_dequant_cached_launch(
        static_cast<cublasHandle_t>(cublas_handle_),
        input.ptr<f32>(), input.size(),
        weight.ptr<i8>(), weight.size(),
        scale_tensor_.ptr<f32>(), scale_tensor_.size(),
        quant_params_.group_size,
        output.ptr<f32>(), output.size(),
        batch_size, K, M,
        reinterpret_cast<f32**>(&weight_fp32_cache_),
        nullptr);
  }
}
#endif

Result<void> MatMulOp::quantize_weight(i32 group_size) {
  // Check if already quantized
  if (is_quantized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "Weight is already quantized");
  }

  // Check if weight is set
  if (weights_.empty() || weights_[0].size() == 0) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "No weight to quantize (call set_weight first)");
  }

  Tensor& weight_fp32 = weights_[0];
  DeviceType original_device = weight_fp32.device();

  // Move weight to CPU if needed (quantize_tensor only supports CPU)
  Tensor weight_cpu;
  if (original_device == DeviceType::CUDA) {
    auto to_cpu_result = weight_fp32.to(DeviceType::CPU);
    if (!to_cpu_result) {
      return Err<void>(to_cpu_result.error());
    }
    weight_cpu = std::move(to_cpu_result.value());
  } else {
    weight_cpu = std::move(weight_fp32);
  }

  // Quantize the weight tensor (on CPU)
  auto quant_result = quantize_tensor(weight_cpu, group_size);
  if (!quant_result) {
    return Err<void>(quant_result.error());
  }

  auto [weight_quant, params] = std::move(quant_result.value());

  // Move quantized weight back to original device if needed
  if (original_device == DeviceType::CUDA) {
    auto to_gpu_result = weight_quant.to(DeviceType::CUDA);
    if (!to_gpu_result) {
      return Err<void>(to_gpu_result.error());
    }
    weight_quant = std::move(to_gpu_result.value());
  }

  // Switch to quantized mode BEFORE calling set_quantized_weight
  // (set_quantized_weight checks is_quantized_)
  is_quantized_ = true;

  // Set quantized weight (this will handle device placement)
  auto set_result = set_quantized_weight(std::move(weight_quant), std::move(params));
  if (!set_result) {
    // Revert is_quantized_ on failure
    is_quantized_ = false;
    return set_result;
  }

  return Ok();
}

}  // namespace photon
