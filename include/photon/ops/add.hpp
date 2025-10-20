/**
 * @file add.hpp
 * @brief Element-wise addition operator for residual connections
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_ADD_HPP
#define PHOTON_OPS_ADD_HPP

#include "operator.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

/**
 * @class AddOp
 * @brief Element-wise tensor addition operator
 *
 * Computes element-wise sum of two tensors:
 * ```
 * output[i] = input1[i] + input2[i]
 * ```
 *
 * This operator is commonly used for:
 * - Residual connections in transformer models
 * - Skip connections in neural networks
 * - Combining feature representations
 *
 * Features:
 * - Broadcasting support for compatible shapes
 * - In-place operation support (output can alias input)
 * - Naive and Eigen-optimized implementations
 * - SIMD vectorization via Eigen
 *
 * Example:
 * ```cpp
 * AddOp add_op;
 * add_op.init();
 *
 * auto x = Tensor::create({1024}, DataType::Float32).value();
 * auto residual = Tensor::create({1024}, DataType::Float32).value();
 * auto output = Tensor::create({1024}, DataType::Float32).value();
 *
 * // Residual connection: output = x + residual
 * add_op.forward(x, residual, output);
 * ```
 */
class AddOp : public OperatorBase<AddOp> {
 public:
  /**
   * @brief Construct Add operator
   *
   * @param use_naive Use naive implementation (for benchmarking)
   */
  explicit AddOp(bool use_naive = false);

  /**
   * @brief Forward pass: output = input1 + input2
   *
   * @param input1 First input tensor
   * @param input2 Second input tensor
   * @param output Output tensor (can alias input1 or input2 for in-place)
   * @return Result<void> Success or error
   *
   * Requirements:
   * - input1 and input2 must have the same shape
   * - output must have the same shape as inputs
   * - All tensors must be on the same device
   * - All tensors must have the same dtype
   */
  Result<void> forward(const Tensor& input1, const Tensor& input2, Tensor& output);

  [[nodiscard]] constexpr bool use_naive() const noexcept { return use_naive_; }

  /**
   * @brief Initialize the operator
   */
  Result<void> init_impl() {
    // No special initialization needed for Add
    return Ok();
  }

  /**
   * @brief Get operator name
   */
  static constexpr std::string_view name_impl() noexcept {
    return "Add";
  }

  /**
   * @brief Get operator category
   */
  static constexpr OpCategory category_impl() noexcept {
    return OpCategory::Elementwise;
  }

 private:
  bool use_naive_;  // Use naive implementation
};

// Verify AddOp satisfies the BinaryOperator concept
static_assert(BinaryOperator<AddOp>, "AddOp must satisfy BinaryOperator concept");

}  // namespace photon

#endif  // PHOTON_OPS_ADD_HPP
