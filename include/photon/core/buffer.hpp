/**
 * @file buffer.hpp
 * @brief Memory buffer abstraction with device-agnostic interface
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 *
 * This file provides a Buffer class that manages raw memory on different devices.
 * Buffers are the foundation for Tensors and provide ownership semantics for
 * device memory.
 */

#ifndef PHOTON_CORE_BUFFER_HPP
#define PHOTON_CORE_BUFFER_HPP

#include <cstring>
#include <memory>
#include <span>
#include <utility>

#include "allocator.hpp"
#include "error.hpp"
#include "types.hpp"

namespace photon {

// ============================================================================
// Buffer Class
// ============================================================================

/**
 * @class Buffer
 * @brief RAII wrapper for device memory
 *
 * Buffer manages a contiguous block of memory on a specific device.
 * It provides ownership semantics and automatic deallocation.
 *
 * Key features:
 * - Move-only semantics (no implicit copying)
 * - Automatic memory management via RAII
 * - Device-agnostic interface
 * - Type-safe memory views via std::span
 */
class Buffer {
 public:
  /**
   * @brief Default constructor creates an empty buffer
   */
  Buffer() = default;

  /**
   * @brief Create a buffer with specified size and device
   *
   * @param size Number of bytes to allocate
   * @param device Device to allocate memory on
   * @param alignment Memory alignment requirement
   * @return Result containing Buffer or error
   */
  [[nodiscard]] static Result<Buffer> create(
      usize size, DeviceType device = DeviceType::CPU, usize alignment = 64) {
    if (size == 0) {
      return Err<Buffer>(ErrorCode::InvalidArgument,
                         "Cannot create buffer with zero size");
    }

    if (device == DeviceType::CPU) {
      CPUAllocator allocator;
      auto alloc_result = allocator.allocate(size, alignment);

      if (!alloc_result) {
        return Err<Buffer>(std::move(alloc_result.error()));
      }

      return Ok(Buffer(alloc_result.value(), size, device, alignment));
    }

#ifdef PHOTON_USE_CUDA
    if (device == DeviceType::CUDA) {
      CUDAAllocator allocator;
      auto alloc_result = allocator.allocate(size, alignment);

      if (!alloc_result) {
        return Err<Buffer>(std::move(alloc_result.error()));
      }

      return Ok(Buffer(alloc_result.value(), size, device, alignment));
    }
#endif

    return Err<Buffer>(ErrorCode::InvalidArgument,
                       "Unsupported device type");
  }

  /**
   * @brief Destructor - automatically frees memory
   */
  ~Buffer() { free(); }

  // Disable copy (Buffer owns memory)
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  /**
   * @brief Move constructor
   */
  Buffer(Buffer&& other) noexcept
      : data_(other.data_),
        size_(other.size_),
        device_(other.device_),
        alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  /**
   * @brief Move assignment operator
   */
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      free();  // Free current memory

      data_ = other.data_;
      size_ = other.size_;
      device_ = other.device_;
      alignment_ = other.alignment_;

      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /**
   * @brief Get raw pointer to buffer data
   */
  [[nodiscard]] void* data() noexcept { return data_; }

  [[nodiscard]] const void* data() const noexcept { return data_; }

  /**
   * @brief Get typed pointer to buffer data
   */
  template <typename T>
  [[nodiscard]] T* data_as() noexcept {
    return static_cast<T*>(data_);
  }

  template <typename T>
  [[nodiscard]] const T* data_as() const noexcept {
    return static_cast<const T*>(data_);
  }

  /**
   * @brief Get buffer size in bytes
   */
  [[nodiscard]] usize size() const noexcept { return size_; }

  /**
   * @brief Get number of elements of type T
   */
  template <typename T>
  [[nodiscard]] usize num_elements() const noexcept {
    return size_ / sizeof(T);
  }

  /**
   * @brief Get device type
   */
  [[nodiscard]] DeviceType device() const noexcept { return device_; }

  /**
   * @brief Get alignment
   */
  [[nodiscard]] usize alignment() const noexcept { return alignment_; }

  /**
   * @brief Check if buffer is empty
   */
  [[nodiscard]] bool empty() const noexcept {
    return data_ == nullptr || size_ == 0;
  }

  /**
   * @brief Boolean conversion (false if empty)
   */
  [[nodiscard]] explicit operator bool() const noexcept { return !empty(); }

  /**
   * @brief Get a typed span view of the buffer (CPU only)
   *
   * @tparam T Element type
   * @return std::span<T> providing safe array-like access
   */
  template <typename T>
  [[nodiscard]] std::span<T> as_span() {
    if (device_ != DeviceType::CPU) {
      // In debug mode, we could throw or log an error
      // For now, return empty span
      return {};
    }
    return std::span<T>(data_as<T>(), num_elements<T>());
  }

  template <typename T>
  [[nodiscard]] std::span<const T> as_span() const {
    if (device_ != DeviceType::CPU) {
      return {};
    }
    return std::span<const T>(data_as<T>(), num_elements<T>());
  }

  /**
   * @brief Fill buffer with zeros
   *
   * @return Result indicating success or failure
   */
  [[nodiscard]] Result<void> zero() {
    if (empty()) {
      return Ok();
    }

    if (device_ == DeviceType::CPU) {
      std::memset(data_, 0, size_);
      return Ok();
    }

#ifdef PHOTON_USE_CUDA
    if (device_ == DeviceType::CUDA) {
      // cudaMemset implementation
      cudaError_t err = cudaMemset(data_, 0, size_);
      if (err != cudaSuccess) {
        return Err<void>(ErrorCode::CudaError,
                         std::string("cudaMemset failed: ") +
                             cudaGetErrorString(err));
      }
      return Ok();
    }
#endif

    return Err<void>(ErrorCode::InvalidArgument, "Unsupported device");
  }

  /**
   * @brief Copy data from another buffer
   *
   * @param src Source buffer
   * @return Result indicating success or failure
   */
  [[nodiscard]] Result<void> copy_from(const Buffer& src) {
    if (src.size_ != size_) {
      return Err<void>(ErrorCode::InvalidArgument,
                       "Source and destination buffers must have same size");
    }

    if (empty() || src.empty()) {
      return Ok();
    }

    // CPU to CPU
    if (device_ == DeviceType::CPU && src.device_ == DeviceType::CPU) {
      std::memcpy(data_, src.data_, size_);
      return Ok();
    }

#ifdef PHOTON_USE_CUDA
    // CUDA memory copy
    cudaMemcpyKind kind;
    if (device_ == DeviceType::CPU && src.device_ == DeviceType::CUDA) {
      kind = cudaMemcpyDeviceToHost;
    } else if (device_ == DeviceType::CUDA &&
               src.device_ == DeviceType::CPU) {
      kind = cudaMemcpyHostToDevice;
    } else if (device_ == DeviceType::CUDA &&
               src.device_ == DeviceType::CUDA) {
      kind = cudaMemcpyDeviceToDevice;
    } else {
      return Err<void>(ErrorCode::InvalidArgument,
                       "Unsupported device combination");
    }

    cudaError_t err = cudaMemcpy(data_, src.data_, size_, kind);
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                       std::string("cudaMemcpy failed: ") +
                           cudaGetErrorString(err));
    }
    return Ok();
#else
    return Err<void>(ErrorCode::InvalidArgument,
                     "CUDA support not enabled");
#endif
  }

  /**
   * @brief Clone this buffer (creates a deep copy)
   *
   * @return Result containing cloned Buffer or error
   */
  [[nodiscard]] Result<Buffer> clone() const {
    if (empty()) {
      return Ok(Buffer());
    }

    auto buffer_result = Buffer::create(size_, device_, alignment_);
    if (!buffer_result) {
      return Err<Buffer>(std::move(buffer_result.error()));
    }

    Buffer new_buffer = std::move(buffer_result.value());
    auto copy_result = new_buffer.copy_from(*this);

    if (!copy_result) {
      return Err<Buffer>(std::move(copy_result.error()));
    }

    return Ok(std::move(new_buffer));
  }

 private:
  /**
   * @brief Private constructor (use create() factory method)
   */
  Buffer(void* data, usize size, DeviceType device, usize alignment)
      : data_(data), size_(size), device_(device), alignment_(alignment) {}

  /**
   * @brief Free allocated memory
   */
  void free() {
    if (data_ == nullptr) {
      return;
    }

    if (device_ == DeviceType::CPU) {
      CPUAllocator allocator;
      auto result = allocator.deallocate(data_, size_);
      // Silently ignore deallocation errors in destructor
      (void)result;
    }
#ifdef PHOTON_USE_CUDA
    else if (device_ == DeviceType::CUDA) {
      CUDAAllocator allocator;
      auto result = allocator.deallocate(data_, size_);
      (void)result;
    }
#endif

    data_ = nullptr;
    size_ = 0;
  }

  void* data_ = nullptr;
  usize size_ = 0;
  DeviceType device_ = DeviceType::CPU;
  usize alignment_ = 64;
};

}  // namespace photon

#endif  // PHOTON_CORE_BUFFER_HPP
