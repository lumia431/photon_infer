/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file allocator.cu
 * @brief CUDA memory allocator implementation with memory pooling
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 *
 * This implementation strictly follows standard memory pool design at:
 * demos/memory pool design
 *
 * Key features (from standard):
 * - Separates big buffers (>1MB) and regular buffers (<=1MB)
 * - Reuses memory blocks to reduce cudaMalloc/cudaFree overhead
 * - Automatic cleanup when idle memory exceeds threshold (1GB)
 */

#ifdef PHOTON_USE_CUDA

#include "photon/core/allocator.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>

namespace photon {

// Using standard threshold: 1MB for big buffers
constexpr usize kBigBufferThreshold = 1024 * 1024;

// Using standard cleanup threshold: 1GB
constexpr usize kCleanupThreshold = 1024UL * 1024UL * 1024UL;

Result<void*> CUDAAllocator::allocate(usize size, [[maybe_unused]] usize alignment) {
  if (size == 0) {
    return Err<void*>(ErrorCode::InvalidArgument,
                      "Cannot allocate zero bytes");
  }

  // Get current device ID (using standard)
  i32 id = -1;
  cudaError_t state = cudaGetDevice(&id);
  if (state != cudaSuccess) {
    LOG(ERROR) << "cudaGetDevice failed: " << cudaGetErrorString(state);
    return Err<void*>(ErrorCode::CudaError, "cudaGetDevice failed");
  }

  // Big buffer path (>1MB) - using standard line-by-line
  if (size > kBigBufferThreshold) {
    auto& big_buffers = big_buffers_map_[id];
    i32 sel_id = -1;

    // Find suitable buffer: size >= requested, not busy, size difference < 1MB
    for (usize i = 0; i < big_buffers.size(); ++i) {
      if (big_buffers[i].byte_size >= size &&
          !big_buffers[i].busy &&
          big_buffers[i].byte_size - size < kBigBufferThreshold) {
        // Select the smallest suitable buffer
        if (sel_id == -1 ||
            big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = static_cast<i32>(i);
        }
      }
    }

    // Reuse existing buffer if found
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return Ok(big_buffers[sel_id].data);
    }

    // No suitable buffer found, allocate new one
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, size);
    if (state != cudaSuccess) {
      LOG(ERROR) << "Error: CUDA error when allocating " << (size >> 20)
                 << " MB memory! maybe there's no enough memory left on device.";
      return Err<void*>(ErrorCode::CudaOutOfMemory,
                        "cudaMalloc failed for big buffer");
    }
    big_buffers.emplace_back(ptr, size, true);
    return Ok(ptr);
  }

  // Regular buffer path (<=1MB) - using standard line-by-line
  auto& cuda_buffers = cuda_buffers_map_[id];

  // Find first fit buffer
  for (usize i = 0; i < cuda_buffers.size(); ++i) {
    if (cuda_buffers[i].byte_size >= size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return Ok(cuda_buffers[i].data);
    }
  }

  // No suitable buffer found, allocate new one
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, size);
  if (state != cudaSuccess) {
    LOG(ERROR) << "Error: CUDA error when allocating " << (size >> 20)
               << " MB memory! maybe there's no enough memory left on device.";
    return Err<void*>(ErrorCode::CudaOutOfMemory,
                      "cudaMalloc failed for regular buffer");
  }
  cuda_buffers.emplace_back(ptr, size, true);
  return Ok(ptr);
}

Result<void> CUDAAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) {
  // Using standard: silently return for null pointer
  if (!ptr) {
    return Ok();
  }

  if (cuda_buffers_map_.empty()) {
    return Ok();
  }

  cudaError_t state = cudaSuccess;

  // Cleanup phase: free idle memory if threshold exceeded (using standard line-by-line)
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > kCleanupThreshold) {
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;

      // Only keep busy buffers
      for (usize i = 0; i < cuda_buffers.size(); ++i) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          if (state != cudaSuccess) {
            LOG(ERROR) << "Error: CUDA error when release memory on device " << it.first;
          }
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }

      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  // Mark buffer as non-busy (using standard line-by-line)
  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (usize i = 0; i < cuda_buffers.size(); ++i) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return Ok();
      }
    }

    // Check big buffers
    auto& big_buffers = big_buffers_map_[it.first];
    for (usize i = 0; i < big_buffers.size(); ++i) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return Ok();
      }
    }
  }

  // Buffer not found in pool, free directly (using standard)
  state = cudaFree(ptr);
  if (state != cudaSuccess) {
    LOG(ERROR) << "Error: CUDA error when release memory on device";
    return Err<void>(ErrorCode::CudaError, "cudaFree failed");
  }

  return Ok();
}

Result<void> CUDAAllocator::set_device(i32 device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaInvalidDevice,
                     std::string("cudaSetDevice failed: ") +
                         cudaGetErrorString(err));
  }

  device_id_ = device_id;
  return Ok();
}

}  // namespace photon

#endif  // PHOTON_USE_CUDA
