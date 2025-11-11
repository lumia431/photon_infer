/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file thrust_allocator.cu
 * @brief Implementation of Thrust-based CUDA allocator
 */

#ifdef PHOTON_USE_CUDA

#include "photon/core/thrust_allocator.hpp"
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace photon {

Result<void*> ThrustAllocator::allocate(usize size, [[maybe_unused]] usize alignment) {
  if (size == 0) {
    return Err<void*>(ErrorCode::InvalidArgument, "Cannot allocate zero bytes");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Use cudaMalloc directly - Thrust's caching is at a higher level
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc failed for " << (size / (1024.0 * 1024.0))
                 << " MB: " << cudaGetErrorString(err);
      return Err<void*>(ErrorCode::CudaOutOfMemory,
                        std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }

    // Track allocation
    allocations_[ptr] = size;
    stats_.total_allocated += size;
    stats_.current_usage += size;
    if (stats_.current_usage > stats_.peak_usage) {
      stats_.peak_usage = stats_.current_usage;
    }

    return Ok(ptr);

  } catch (const std::exception& e) {
    return Err<void*>(ErrorCode::CudaError,
                      std::string("Thrust allocation failed: ") + e.what());
  }
}

Result<void> ThrustAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) {
  if (!ptr) {
    return Ok();
  }

  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Find allocation size
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      cudaFree(ptr);
      return Ok();
    }

    usize alloc_size = it->second;
    allocations_.erase(it);

    // Free memory
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(err);
      return Err<void>(ErrorCode::CudaError,
                      std::string("cudaFree failed: ") + cudaGetErrorString(err));
    }

    // Update stats
    stats_.total_freed += alloc_size;
    stats_.current_usage -= alloc_size;

    return Ok();

  } catch (const std::exception& e) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("Thrust deallocation failed: ") + e.what());
  }
}

Result<void> ThrustAllocator::set_device(i32 device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaInvalidDevice,
                     std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
  }

  device_id_ = device_id;
  return Ok();
}

ThrustAllocator::MemoryStats ThrustAllocator::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

} // namespace photon

#endif // PHOTON_USE_CUDA
