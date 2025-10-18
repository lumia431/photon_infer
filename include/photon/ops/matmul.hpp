/**
 * @file matmul.hpp
 * @brief Matrix multiplication operator
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_MATMUL_HPP
#define PHOTON_OPS_MATMUL_HPP

#include "operator.hpp"
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
   */
  explicit MatMulOp(i32 input_dim, i32 output_dim, bool use_naive = false)
      : input_dim_(input_dim), output_dim_(output_dim), use_naive_(use_naive) {
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

    if (weight.dtype() != DataType::Float32) {
      return Err<void>(ErrorCode::InvalidDtype,
                      "MatMul weight must be Float32");
    }

    if (weight.device() != device_) {
      return Err<void>(ErrorCode::DeviceMismatch,
                      "Weight device does not match operator device");
    }

    weights_[0] = std::move(weight);
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

 private:
  i32 input_dim_;      ///< Input feature dimension (N)
  i32 output_dim_;     ///< Output feature dimension (M)
  bool use_naive_;     ///< Use naive implementation flag

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

#endif  // PHOTON_OPS_MATMUL_HPP
