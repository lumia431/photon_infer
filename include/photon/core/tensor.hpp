/**
 * @file tensor.hpp
 * @brief Multi-dimensional tensor abstraction for deep learning
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-16
 *
 * This file provides a Tensor class that represents multi-dimensional arrays
 * with shape management, type safety, and device abstraction. Tensors are built
 * on top of Buffers and integrate with Eigen for efficient linear algebra.
 *
 * Design Philosophy (inspired by KuiperInfer):
 * - Tensor manages memory, shape, and device information
 * - Use Eigen::Map for zero-copy matrix/vector views in operators
 * - Separation of concerns: storage (Tensor) vs computation (Eigen)
 */

#ifndef PHOTON_CORE_TENSOR_HPP
#define PHOTON_CORE_TENSOR_HPP

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "buffer.hpp"
#include "error.hpp"
#include "types.hpp"

namespace photon {

// ============================================================================
// Tensor Class
// ============================================================================

/**
 * @class Tensor
 * @brief Multi-dimensional array with Eigen integration
 *
 * Tensor provides:
 * - Memory management via Buffer
 * - Shape and stride tracking
 * - Device-agnostic storage (CPU/CUDA)
 * - Zero-copy Eigen views for computation
 *
 * Usage pattern:
 * ```cpp
 * // Create tensor
 * auto tensor = Tensor::create({2, 3}, DataType::Float32);
 *
 * // Get Eigen map for computation (in operator implementations)
 * auto mat_map = tensor.matrix_map<float>();
 * mat_map = mat_map * 2.0f;  // Uses Eigen's optimized operations
 * ```
 *
 * Design notes:
 * - Tensor owns memory (move-only semantics)
 * - Eigen::Map provides zero-copy views (like Armadillo in KuiperInfer)
 * - Shape is stored in row-major order (C-style, compatible with NumPy)
 */
class Tensor {
 public:
  /**
   * @brief Default constructor creates an empty tensor
   */
  Tensor() = default;

  /**
   * @brief Create a tensor with specified shape and data type
   *
   * @param dims Dimensions of the tensor
   * @param dtype Data type of elements
   * @param device Device to allocate memory on
   * @param need_alloc Whether to allocate memory immediately
   * @param alloc Custom allocator (optional)
   * @param ptr External memory pointer (if not allocating)
   * @return Result containing Tensor or error
   */
  [[nodiscard]] static Result<Tensor> create(
      std::vector<int32_t> dims, DataType dtype,
      DeviceType device = DeviceType::CPU, bool need_alloc = true);

  /**
   * @brief Create a tensor and initialize with zeros
   */
  [[nodiscard]] static Result<Tensor> zeros(
      std::vector<int32_t> dims, DataType dtype,
      DeviceType device = DeviceType::CPU);

  /**
   * @brief Create a 1D tensor from raw data (copies data)
   */
  template <typename T>
  [[nodiscard]] static Result<Tensor> from_vector(
      const std::vector<T>& data, DeviceType device = DeviceType::CPU);

  /**
   * @brief Create a 2D tensor from raw data
   */
  template <typename T>
  [[nodiscard]] static Result<Tensor> from_matrix(
      int32_t rows, int32_t cols, const std::vector<T>& data,
      DeviceType device = DeviceType::CPU);

  // Disable copy (Tensor owns memory)
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  /**
   * @brief Move constructor
   */
  Tensor(Tensor&& other) noexcept = default;

  /**
   * @brief Move assignment
   */
  Tensor& operator=(Tensor&& other) noexcept = default;

  // ========== Getters ==========

  /**
   * @brief Get number of dimensions
   */
  [[nodiscard]] int32_t ndim() const noexcept { return static_cast<int32_t>(dims_.size()); }

  /**
   * @brief Get dimension at index
   */
  [[nodiscard]] int32_t dim(int32_t index) const {
    if (index < 0 || index >= ndim()) {
      return 0;
    }
    return dims_[index];
  }

  /**
   * @brief Get all dimensions
   */
  [[nodiscard]] const std::vector<int32_t>& dims() const noexcept {
    return dims_;
  }

  /**
   * @brief Get data type
   */
  [[nodiscard]] DataType dtype() const noexcept { return dtype_; }

  /**
   * @brief Get device type
   */
  [[nodiscard]] DeviceType device() const noexcept { return device_; }

  /**
   * @brief Get total number of elements
   */
  [[nodiscard]] usize size() const noexcept { return size_; }

  /**
   * @brief Get size in bytes
   */
  [[nodiscard]] usize byte_size() const noexcept {
    return size_ * data_type_size(dtype_);
  }

  /**
   * @brief Check if tensor is empty
   */
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  /**
   * @brief Get strides (row-major)
   */
  [[nodiscard]] std::vector<usize> strides() const;

  // ========== Raw Pointer Access ==========

  /**
   * @brief Get raw data pointer
   */
  [[nodiscard]] void* data() noexcept { return buffer_.data(); }

  /**
   * @brief Get const raw data pointer
   */
  [[nodiscard]] const void* data() const noexcept { return buffer_.data(); }

  /**
   * @brief Get typed data pointer
   */
  template <typename T>
  [[nodiscard]] T* ptr() noexcept {
    return buffer_.data_as<T>();
  }

  /**
   * @brief Get const typed data pointer
   */
  template <typename T>
  [[nodiscard]] const T* ptr() const noexcept {
    return buffer_.data_as<T>();
  }

  /**
   * @brief Get typed data pointer with offset
   */
  template <typename T>
  [[nodiscard]] T* ptr(int64_t offset) noexcept {
    return buffer_.data_as<T>() + offset;
  }

  /**
   * @brief Get const typed data pointer with offset
   */
  template <typename T>
  [[nodiscard]] const T* ptr(int64_t offset) const noexcept {
    return buffer_.data_as<T>() + offset;
  }

  /**
   * @brief Index into tensor (flat index)
   */
  template <typename T>
  [[nodiscard]] T& index(int64_t offset) {
    return buffer_.data_as<T>()[offset];
  }

  /**
   * @brief Const index into tensor
   */
  template <typename T>
  [[nodiscard]] const T& index(int64_t offset) const {
    return buffer_.data_as<T>()[offset];
  }

  // ========== Eigen Integration (Zero-Copy Views) ==========

  /**
   * @brief Get Eigen::Map as 1D vector (CPU only)
   *
   * Example:
   * ```cpp
   * auto vec = tensor.vector_map<float>();
   * vec = vec * 2.0f;  // In-place scaling
   * ```
   */
  template <typename T>
  [[nodiscard]] Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vector_map() {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(ptr<T>(), size_);
  }

  /**
   * @brief Get const Eigen::Map as 1D vector
   */
  template <typename T>
  [[nodiscard]] Eigen::Map<const Eigen::Matrix<std::remove_const_t<T>, Eigen::Dynamic, 1>>
  vector_map() const {
    using ScalarT = std::remove_const_t<T>;
    return Eigen::Map<const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>>(ptr<T>(), size_);
  }

  /**
   * @brief Get Eigen::Map as 2D matrix (row-major, CPU only)
   *
   * Example:
   * ```cpp
   * auto mat = tensor.matrix_map<float>();  // Requires rank == 2
   * mat = mat * other_mat;  // Matrix multiplication
   * ```
   */
  template <typename T>
  [[nodiscard]] Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  matrix_map() {
    if (ndim() != 2) {
      // Return empty map for non-2D tensors
      return Eigen::Map<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          nullptr, 0, 0);
    }
    return Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        ptr<T>(), dims_[0], dims_[1]);
  }

  /**
   * @brief Get const Eigen::Map as 2D matrix
   */
  template <typename T>
  [[nodiscard]] Eigen::Map<
      const Eigen::Matrix<std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  matrix_map() const {
    using ScalarT = std::remove_const_t<T>;
    if (ndim() != 2) {
      return Eigen::Map<const Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic,
                                            Eigen::RowMajor>>(nullptr, 0, 0);
    }
    return Eigen::Map<
        const Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        ptr<T>(), dims_[0], dims_[1]);
  }

  // ========== Operations ==========

  /**
   * @brief Reshape tensor (must have same total size)
   */
  [[nodiscard]] Result<void> reshape(const std::vector<int32_t>& new_dims);

  /**
   * @brief Clone tensor (deep copy)
   */
  [[nodiscard]] Result<Tensor> clone() const;

  /**
   * @brief Reset tensor with new shape and type
   */
  void reset(DataType dtype, const std::vector<int32_t>& dims);

  /**
   * @brief Copy data to CPU
   */
  [[nodiscard]] Result<void> to_cpu();

  /**
   * @brief Copy data to CUDA
   */
  [[nodiscard]] Result<void> to_cuda();

  /**
   * @brief Convert tensor to string representation
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief Get underlying buffer
   */
  [[nodiscard]] const Buffer& buffer() const noexcept { return buffer_; }

 private:
  Buffer buffer_;                ///< Underlying memory buffer
  std::vector<int32_t> dims_;    ///< Tensor dimensions
  usize size_ = 0;               ///< Total number of elements
  DataType dtype_ = DataType::Float32;   ///< Element data type
  DeviceType device_ = DeviceType::CPU;  ///< Device where data resides

  /**
   * @brief Private constructor from buffer
   */
  Tensor(Buffer&& buffer, std::vector<int32_t> dims, DataType dtype,
         DeviceType device);

  /**
   * @brief Compute total size from dimensions
   */
  static usize compute_size(const std::vector<int32_t>& dims);
};

// ============================================================================
// Template Implementations
// ============================================================================

template <typename T>
Result<Tensor> Tensor::from_vector(const std::vector<T>& data,
                                   DeviceType device) {
  DataType dtype = cpp_type_to_data_type_v<T>;
  std::vector<int32_t> dims = {static_cast<int32_t>(data.size())};

  auto result = create(dims, dtype, device);
  if (!result) {
    return result;
  }

  Tensor tensor = std::move(result.value());

  // Copy data to tensor (CPU only for now)
  if (device == DeviceType::CPU) {
    std::memcpy(tensor.ptr<T>(), data.data(), data.size() * sizeof(T));
  } else {
    return Err<Tensor>(ErrorCode::NotImplemented,
                      "CUDA data copy not yet implemented");
  }

  return Ok(std::move(tensor));
}

template <typename T>
Result<Tensor> Tensor::from_matrix(int32_t rows, int32_t cols,
                                   const std::vector<T>& data,
                                   DeviceType device) {
  if (data.size() != static_cast<usize>(rows * cols)) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "Data size mismatch: expected " +
                          std::to_string(rows * cols) + ", got " +
                          std::to_string(data.size()));
  }

  DataType dtype = cpp_type_to_data_type_v<T>;
  std::vector<int32_t> dims = {rows, cols};

  auto result = create(dims, dtype, device);
  if (!result) {
    return result;
  }

  Tensor tensor = std::move(result.value());

  if (device == DeviceType::CPU) {
    std::memcpy(tensor.ptr<T>(), data.data(), data.size() * sizeof(T));
  } else {
    return Err<Tensor>(ErrorCode::NotImplemented,
                      "CUDA data copy not yet implemented");
  }

  return Ok(std::move(tensor));
}

}  // namespace photon

#endif  // PHOTON_CORE_TENSOR_HPP
