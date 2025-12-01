/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file operator.hpp
 * @brief Base operator interfaces using Modern C++20 features (CRTP + Concepts)
 * @version 0.1.0
 */


#include <concepts>
#include <string_view>
#include <vector>

#include "photon/core/error.hpp"
#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"

namespace photon {

// ============================================================================
// Operator Type Concepts
// ============================================================================

/**
 * @brief Concept to constrain operator types
 *
 * An operator must provide:
 * - init() method returning Result<void>
 * - name() method returning string_view
 */
template <typename T>
concept Operator = requires(T op) {
  { op.init() } -> std::same_as<Result<void>>;
  { op.name() } -> std::convertible_to<std::string_view>;
};

/**
 * @brief Concept for operators with single input/output forward
 */
template <typename T>
concept UnaryOperator = Operator<T> && requires(T op, const Tensor& in, Tensor& out) {
  { op.forward(in, out) } -> std::same_as<Result<void>>;
};

/**
 * @brief Concept for operators with two inputs and one output
 */
template <typename T>
concept BinaryOperator = Operator<T> && requires(T op, const Tensor& in1, const Tensor& in2, Tensor& out) {
  { op.forward(in1, in2, out) } -> std::same_as<Result<void>>;
};

// ============================================================================
// Operator Categories
// ============================================================================

enum class OpCategory : u8 {
  Unknown = 0,
  Embedding = 1,
  Normalization = 2,
  MatMul = 3,
  Attention = 4,
  Activation = 5,
  Elementwise = 6,
};

constexpr std::string_view op_category_str(OpCategory cat) noexcept {
  switch (cat) {
    case OpCategory::Unknown: return "Unknown";
    case OpCategory::Embedding: return "Embedding";
    case OpCategory::Normalization: return "Normalization";
    case OpCategory::MatMul: return "MatMul";
    case OpCategory::Attention: return "Attention";
    case OpCategory::Activation: return "Activation";
    case OpCategory::Elementwise: return "Elementwise";
    default: return "Unknown";
  }
}

// ============================================================================
// CRTP Base Operator Class
// ============================================================================

/**
 * @brief CRTP base class for operators (static polymorphism)
 *
 * This class uses the Curiously Recurring Template Pattern (CRTP) to achieve
 * static polymorphism without virtual function overhead.
 *
 * Derived classes must implement:
 * - init_impl() -> Result<void>
 * - forward_impl(...) -> Result<void>
 *
 * @tparam Derived The derived operator class
 */
template <typename Derived>
class OperatorBase {
 public:
  /**
   * @brief Initialize the operator
   */
  Result<void> init() {
    if (initialized_) {
      return Ok();
    }
    auto result = static_cast<Derived*>(this)->init_impl();
    if (result) {
      initialized_ = true;
    }
    return result;
  }

  /**
   * @brief Get operator name
   */
  std::string_view name() const noexcept {
    return static_cast<const Derived*>(this)->name_impl();
  }

  /**
   * @brief Get operator category
   */
  OpCategory category() const noexcept {
    return static_cast<const Derived*>(this)->category_impl();
  }

  /**
   * @brief Check if operator is initialized
   */
  [[nodiscard]] bool is_initialized() const noexcept {
    return initialized_;
  }

  /**
   * @brief Get device type
   */
  [[nodiscard]] DeviceType device() const noexcept {
    return device_;
  }

  /**
   * @brief Set device type
   */
  void set_device(DeviceType device) noexcept {
    device_ = device;
  }

  /**
   * @brief Get data type
   */
  [[nodiscard]] DataType dtype() const noexcept {
    return dtype_;
  }

  /**
   * @brief Set data type
   */
  void set_dtype(DataType dtype) noexcept {
    dtype_ = dtype;
  }

 protected:
  OperatorBase() = default;
  ~OperatorBase() = default;

  // Prevent slicing
  OperatorBase(const OperatorBase&) = delete;
  OperatorBase& operator=(const OperatorBase&) = delete;
  OperatorBase(OperatorBase&&) = default;
  OperatorBase& operator=(OperatorBase&&) = default;

  DeviceType device_ = DeviceType::CPU;
  DataType dtype_ = DataType::Float32;
  bool initialized_ = false;
};

// ============================================================================
// Parameterized Operator Base (for layers with weights)
// ============================================================================

/**
 * @brief Base class for operators with parameters/weights
 *
 * This extends OperatorBase with weight management capabilities.
 *
 * @tparam Derived The derived operator class
 */
template <typename Derived>
class ParameterizedOperator : public OperatorBase<Derived> {
 public:
  /**
   * @brief Set a weight tensor by index
   */
  Result<void> set_weight(usize index, Tensor weight) {
    if (index >= weights_.size()) {
      weights_.resize(index + 1);
    }

    // Validate weight device matches operator device
    if (weight.device() != this->device_) {
      return Err<void>(ErrorCode::DeviceMismatch,
                      "Weight device does not match operator device");
    }

    weights_[index] = std::move(weight);
    return Ok();
  }

  /**
   * @brief Get weight tensor by index
   */
  [[nodiscard]] const Tensor& get_weight(usize index) const {
    return weights_[index];
  }

  /**
   * @brief Get mutable weight tensor by index
   */
  [[nodiscard]] Tensor& get_weight(usize index) {
    return weights_[index];
  }

  /**
   * @brief Get number of weight tensors
   */
  [[nodiscard]] usize num_weights() const noexcept {
    return weights_.size();
  }

  /**
   * @brief Check if all weights are set
   */
  [[nodiscard]] bool weights_initialized() const noexcept {
    for (const auto& w : weights_) {
      if (w.empty()) {
        return false;
      }
    }
    return !weights_.empty();
  }

 protected:
  std::vector<Tensor> weights_;
};

}  // namespace photon

