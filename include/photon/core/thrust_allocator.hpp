/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#pragma once

/**
 * @file thrust_allocator.hpp
 * @brief Thrust-based CUDA memory allocator
 * @version 1.0.0
 *
 * This allocator uses Thrust's device_vector for automatic CUDA memory management.
 * Benefits:
 * - Automatic memory management (RAII)
 * - Built-in caching allocator
 * - Proven reliability (CUDA official library)
 * - No manual memory pool management
 */

#ifdef PHOTON_USE_CUDA

#include "photon/core/allocator.hpp"
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace photon {

/**
 * @brief Thrust-based CUDA allocator
 *
 * This allocator leverages Thrust's caching allocator which:
 * - Reuses freed memory automatically
 * - Manages memory pool internally
 * - Provides RAII semantics
 * - Thread-safe by design
 */
class ThrustAllocator {
public:
  ThrustAllocator() = default;

  /**
   * @brief Allocate CUDA memory using Thrust
   *
   * @param size Number of bytes to allocate
   * @param alignment Memory alignment (ignored, Thrust handles this)
   * @return Pointer to allocated memory or error
   */
  [[nodiscard]] Result<void*> allocate(usize size, usize alignment = 256);

  /**
   * @brief Deallocate CUDA memory
   *
   * @param ptr Pointer to memory to free
   * @param size Size of memory block
   * @return Success or error
   */
  [[nodiscard]] Result<void> deallocate(void* ptr, usize size);

  /**
   * @brief Get device type
   */
  [[nodiscard]] constexpr DeviceType device_type() const noexcept {
    return DeviceType::CUDA;
  }

  /**
   * @brief Set CUDA device
   */
  Result<void> set_device(i32 device_id);

  /**
   * @brief Get current device
   */
  [[nodiscard]] i32 device_id() const noexcept { return device_id_; }

  /**
   * @brief Get memory usage statistics
   */
  struct MemoryStats {
    usize total_allocated = 0;
    usize total_freed = 0;
    usize current_usage = 0;
    usize peak_usage = 0;
  };

  [[nodiscard]] MemoryStats get_stats() const;

private:
  i32 device_id_ = 0;

  // Track allocations for statistics
  mutable std::mutex mutex_;
  std::unordered_map<void*, usize> allocations_;
  MemoryStats stats_;
};

static_assert(Allocator<ThrustAllocator>,
              "ThrustAllocator must satisfy Allocator concept");

} // namespace photon

#endif // PHOTON_USE_CUDA
