/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file matmul.hpp
 * @brief Matrix multiplication operator
 * @version 0.1.0
 */


#include "operator.hpp"
#include "photon/core/quant.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// MatMul Operator
// ============================================================================

/**
 * @class MatMulOp
 * @brief Matrix multiplication operator: output = input @ weight
 *
 * This operator implements matrix-vector (GEMV) and matrix-matrix (GEMM)
 * multiplication, which is the core operation in transformer models.
 *
 * Supported operations:
 * - Vector-Matrix: [N] @ [N × M] -> [M]
 * - Matrix-Matrix: [B × N] @ [N × M] -> [B × M]
 *
 * Architecture:
 * - Input: [N] or [batch × N]
 * - Weight: [M × N] (transposed for efficient computation)
 * - Output: [M] or [batch × M]
 *
 * Implementation strategies:
 * 1. Naive: Triple-loop implementation (O(N×M×B))
 * 2. Eigen: Optimized BLAS-like operations (SIMD + threading)
 * 3. Quantized: Int8 weights with dynamic dequantization
 *
 * Example:
 * ```cpp
 * // Create MatMul: input_dim=512, output_dim=256
 * MatMulOp op(512, 256);
 *
 * // Set weight matrix [256 × 512]
 * auto weight = Tensor::create({256, 512}, DataType::Float32).value();
 * op.set_weight(std::move(weight));
 * op.init();
 *
 * // Forward: [512] @ [256×512]^T -> [256]
 * auto input = Tensor::create({512}, DataType::Float32).value();
 * auto output = Tensor::create({256}, DataType::Float32).value();
 * op.forward(input, output);
 * ```
 */
class MatMulOp : public ParameterizedOperator<MatMulOp> {
 public:
  /**
   * @brief Construct MatMul operator
   *
   * @param input_dim Input feature dimension (N)
   * @param output_dim Output feature dimension (M)
   * @param use_naive Use naive implementation (for benchmarking)
   * @param is_quantized Whether to use int8 quantized weights
   */
  explicit MatMulOp(i32 input_dim, i32 output_dim, bool use_naive = false,
                   bool is_quantized = false)
      : input_dim_(input_dim), output_dim_(output_dim), use_naive_(use_naive),
        is_quantized_(is_quantized) {
    weights_.resize(1);
  }

  /**
   * @brief Set weight matrix
   *
   * @param weight Tensor of shape [output_dim × input_dim]
   * @return Result indicating success or error
   */
  Result<void> set_weight(Tensor weight) {
    // Validate weight shape
    if (weight.ndim() != 2) {
      return Err<void>(ErrorCode::InvalidShape,
                      "MatMul weight must be 2D tensor");
    }

    if (weight.dim(0) != output_dim_ || weight.dim(1) != input_dim_) {
      return Err<void>(
          ErrorCode::ShapeMismatch,
          "Weight shape mismatch: expected [" + std::to_string(output_dim_) +
              " × " + std::to_string(input_dim_) + "], got [" +
              std::to_string(weight.dim(0)) + " × " +
              std::to_string(weight.dim(1)) + "]");
    }

    DataType expected_dtype = is_quantized_ ? DataType::Int8 : DataType::Float32;
    if (weight.dtype() != expected_dtype) {
      return Err<void>(ErrorCode::InvalidDtype,
                      "MatMul weight dtype mismatch: expected " +
                      std::string(data_type_str(expected_dtype)) + ", got " +
                      std::string(data_type_str(weight.dtype())));
    }

    // Auto-convert weight to operator's device if needed
    if (weight.device() != device_) {
      auto converted = weight.to(device_);
      if (!converted) {
        return Err<void>(converted.error());
      }
      weight = std::move(converted.value());
    }

    weights_[0] = std::move(weight);
    return Ok();
  }

  /**
   * @brief Set quantized weight matrix with quantization parameters
   *
   * @param weight Quantized int8 tensor of shape [output_dim × input_dim]
   * @param params Quantization parameters
   * @return Result indicating success or error
   */
  Result<void> set_quantized_weight(Tensor weight, QuantParams params) {
    if (!is_quantized_) {
      return Err<void>(ErrorCode::InvalidOperator,
                      "Operator not configured for quantization");
    }

    // Set weight using existing validation
    auto result = set_weight(std::move(weight));
    if (!result) {
      return result;
    }

    // Validate and store quantization parameters
    if (!params.is_valid()) {
      return Err<void>(ErrorCode::InvalidArgument,
                      "Invalid quantization parameters");
    }

    quant_params_ = std::move(params);

    // Create scale tensor on CPU first
    auto scale_tensor_cpu_result = Tensor::from_vector(
        quant_params_.scales, DeviceType::CPU);
    if (!scale_tensor_cpu_result) {
      return Err<void>(scale_tensor_cpu_result.error());
    }

    // Move to device if needed
    if (device_ == DeviceType::CPU) {
      scale_tensor_ = std::move(scale_tensor_cpu_result.value());
    } else {
      auto scale_tensor_gpu_result = scale_tensor_cpu_result.value().to(device_);
      if (!scale_tensor_gpu_result) {
        return Err<void>(scale_tensor_gpu_result.error());
      }
      scale_tensor_ = std::move(scale_tensor_gpu_result.value());
    }

    return Ok();
  }

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    if (!weights_initialized()) {
      return Err<void>(ErrorCode::InvalidOperator,
                      "MatMul weights not set");
    }

    // For quantized operator, ensure scales are set
    if (is_quantized_ && quant_params_.scales.empty()) {
      return Err<void>(ErrorCode::InvalidOperator,
                      "Quantized MatMul requires quantization parameters");
    }

    return Ok();
  }

  /**
   * @brief Forward pass: input @ weight^T -> output
   *
   * Supports:
   * - GEMV: [N] @ [M×N]^T -> [M]
   * - GEMM: [B×N] @ [M×N]^T -> [B×M]
   *
   * @param input Input tensor, shape [input_dim] or [batch × input_dim]
   * @param output Output tensor, shape [output_dim] or [batch × output_dim]
   * @return Result indicating success or error
   */
  Result<void> forward(const Tensor& input, Tensor& output);

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "MatMulOp";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::MatMul;
  }

  /**
   * @brief Get input dimension
   */
  [[nodiscard]] i32 input_dim() const noexcept { return input_dim_; }

  /**
   * @brief Get output dimension
   */
  [[nodiscard]] i32 output_dim() const noexcept { return output_dim_; }

  /**
   * @brief Check if using naive implementation
   */
  [[nodiscard]] bool is_naive() const noexcept { return use_naive_; }

  /**
   * @brief Check if using quantized weights
   */
  [[nodiscard]] bool is_quantized() const noexcept { return is_quantized_; }

  /**
   * @brief Get quantization parameters (if quantized)
   */
  [[nodiscard]] const QuantParams& quant_params() const noexcept {
    return quant_params_;
  }

  /**
   * @brief Quantize existing FP32 weight to INT8 in-place
   *
   * Converts the current FP32 weight to INT8 quantized format.
   * The operator will be switched to quantized mode after this call.
   *
   * @param group_size Group size for quantization (default 128)
   * @return Result<void> Success or error
   */
  Result<void> quantize_weight(i32 group_size = 128);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief Set cuBLAS handle for FP16 Tensor Core optimization
   *
   * @param handle cuBLAS handle (must remain valid during operator lifetime)
   */
  void set_cublas_handle(void* handle) { cublas_handle_ = handle; }

  /**
   * @brief Destructor - cleanup weight cache
   */
  ~MatMulOp() {
    if (weight_fp32_cache_ != nullptr) {
      cudaFree(weight_fp32_cache_);
      weight_fp32_cache_ = nullptr;
    }
  }
#endif

 private:
  i32 input_dim_;      ///< Input feature dimension (N)
  i32 output_dim_;     ///< Output feature dimension (M)
  bool use_naive_;     ///< Use naive implementation flag
  bool is_quantized_;  ///< Use int8 quantized weights flag

  // Quantization members
  QuantParams quant_params_;  ///< Quantization parameters (scales, group_size)
  Tensor scale_tensor_;       ///< Scale tensor on device for kernel usage

#ifdef PHOTON_USE_CUDA
  // Dequantized weight cache for cuBLAS optimization
  void* weight_fp32_cache_ = nullptr;  ///< Cached FP32 dequantized weights
  void* cublas_handle_ = nullptr;      ///< cuBLAS handle (owned by model)
#endif

  /**
   * @brief CPU forward implementation
   */
  Result<void> forward_cpu(const Tensor& input, Tensor& output);

#ifdef PHOTON_USE_CUDA
  /**
   * @brief CUDA forward implementation
   */
  Result<void> forward_cuda(const Tensor& input, Tensor& output);
#endif
};

// Verify MatMulOp satisfies Operator concept
static_assert(Operator<MatMulOp>, "MatMulOp must satisfy Operator concept");
static_assert(UnaryOperator<MatMulOp>, "MatMulOp must satisfy UnaryOperator concept");

}  // namespace photon

