/**
 * @file allocator.hpp
 * @brief Memory allocator abstraction for different devices
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 *
 * This file provides a unified interface for memory allocation across different
 * devices (CPU, CUDA). It uses modern C++20 features like concepts to ensure
 * type safety and compile-time polymorphism.
 */

#ifndef PHOTON_CORE_ALLOCATOR_HPP
#define PHOTON_CORE_ALLOCATOR_HPP

#include <cstdlib>
#include <memory>
#include <new>

#include "error.hpp"
#include "types.hpp"

namespace photon {

// ============================================================================
// Allocator Concept
// ============================================================================

/**
 * @concept Allocator
 * @brief Concept that defines requirements for memory allocators
 *
 * An Allocator must provide:
 * - allocate(size, alignment) -> Result<void*>
 * - deallocate(ptr, size) -> Result<void>
 * - device_type() -> DeviceType
 */
template <typename T>
concept Allocator = requires(T alloc, usize size, usize alignment, void* ptr) {
  { alloc.allocate(size, alignment) } -> std::same_as<Result<void*>>;
  { alloc.deallocate(ptr, size) } -> std::same_as<Result<void>>;
  { alloc.device_type() } -> std::same_as<DeviceType>;
};

// ============================================================================
// CPU Allocator
// ============================================================================

/**
 * @class CPUAllocator
 * @brief Aligned memory allocator for CPU
 *
 * This allocator provides cache-line aligned memory allocation on CPU
 * using std::aligned_alloc.
 */
class CPUAllocator {
 public:
  /**
   * @brief Default cache line size (64 bytes for most modern CPUs)
   */
  static constexpr usize kDefaultAlignment = 64;

  /**
   * @brief Allocate aligned memory on CPU
   *
   * @param size Number of bytes to allocate
   * @param alignment Alignment requirement (must be power of 2)
   * @return Result containing pointer to allocated memory, or error
   */
  [[nodiscard]] Result<void*> allocate(usize size,
                                       usize alignment = kDefaultAlignment) {
    // Check alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      return Err<void*>(ErrorCode::InvalidAlignment,
                        "Alignment must be power of 2");
    }

    // Check size is not zero
    if (size == 0) {
      return Err<void*>(ErrorCode::InvalidArgument,
                        "Cannot allocate zero bytes");
    }

    // Align size to alignment boundary
    size = (size + alignment - 1) & ~(alignment - 1);

#if defined(_WIN32)
    // Windows: use _aligned_malloc
    void* ptr = _aligned_malloc(size, alignment);
#else
    // POSIX: use aligned_alloc
    void* ptr = std::aligned_alloc(alignment, size);
#endif

    if (ptr == nullptr) {
      return Err<void*>(ErrorCode::OutOfMemory,
                        "Failed to allocate " + std::to_string(size) +
                            " bytes");
    }

    return Ok(ptr);
  }

  /**
   * @brief Deallocate memory previously allocated by this allocator
   *
   * @param ptr Pointer to memory to deallocate
   * @param size Size of the allocation (unused but kept for interface
   * consistency)
   * @return Result indicating success or failure
   */
  [[nodiscard]] Result<void> deallocate(void* ptr,
                                        [[maybe_unused]] usize size) const {
    if (ptr == nullptr) {
      return Err<void>(ErrorCode::NullPointer,
                       "Cannot deallocate null pointer");
    }

#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif

    return Ok();
  }

  /**
   * @brief Get the device type for this allocator
   */
  [[nodiscard]] constexpr DeviceType device_type() const noexcept {
    return DeviceType::CPU;
  }

  /**
   * @brief Get the default alignment used by this allocator
   */
  [[nodiscard]] constexpr usize default_alignment() const noexcept {
    return kDefaultAlignment;
  }
};

// Verify CPUAllocator satisfies Allocator concept
static_assert(Allocator<CPUAllocator>, "CPUAllocator must satisfy Allocator concept");

// ============================================================================
// CUDA Allocator (Forward Declaration)
// ============================================================================

#ifdef PHOTON_USE_CUDA

/**
 * @class CUDAAllocator
 * @brief Memory allocator for CUDA devices
 *
 * This allocator provides memory allocation on CUDA devices using cudaMalloc.
 * Full implementation is in allocator.cu
 */
class CUDAAllocator {
 public:
  [[nodiscard]] Result<void*> allocate(usize size, usize alignment = 256);

  [[nodiscard]] Result<void> deallocate(void* ptr, usize size);

  [[nodiscard]] constexpr DeviceType device_type() const noexcept {
    return DeviceType::CUDA;
  }

  /**
   * @brief Set the CUDA device to use for allocations
   */
  Result<void> set_device(i32 device_id);

  /**
   * @brief Get the currently selected CUDA device
   */
  [[nodiscard]] i32 device_id() const noexcept { return device_id_; }

 private:
  i32 device_id_ = 0;
};

static_assert(Allocator<CUDAAllocator>, "CUDAAllocator must satisfy Allocator concept");

#endif  // PHOTON_USE_CUDA

// ============================================================================
// Unique Pointer with Custom Deleter
// ============================================================================

/**
 * @class AllocatorDeleter
 * @brief Custom deleter for unique_ptr that uses an Allocator
 *
 * This allows us to use std::unique_ptr with our custom allocators.
 */
template <Allocator A>
class AllocatorDeleter {
 public:
  explicit AllocatorDeleter(A allocator, usize size)
      : allocator_(std::move(allocator)), size_(size) {}

  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      auto result = allocator_.deallocate(ptr, size_);
      // In production, handle deallocation errors appropriately
      (void)result;
    }
  }

 private:
  A allocator_;
  usize size_;
};

/**
 * @brief Unique pointer type that uses an allocator for deallocation
 */
template <Allocator A>
using AllocatorUniquePtr = std::unique_ptr<void, AllocatorDeleter<A>>;

/**
 * @brief Create a unique_ptr with memory allocated by an allocator
 *
 * @tparam A Allocator type
 * @param allocator The allocator to use
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement
 * @return Result containing unique_ptr to allocated memory, or error
 */
template <Allocator A>
[[nodiscard]] Result<AllocatorUniquePtr<A>> make_unique_alloc(
    A allocator, usize size, usize alignment) {
  auto alloc_result = allocator.allocate(size, alignment);

  if (!alloc_result) {
    return Err<AllocatorUniquePtr<A>>(std::move(alloc_result.error()));
  }

  void* ptr = alloc_result.value();
  AllocatorDeleter<A> deleter(std::move(allocator), size);
  return Ok(AllocatorUniquePtr<A>(ptr, std::move(deleter)));
}

}  // namespace photon

#endif  // PHOTON_CORE_ALLOCATOR_HPP
