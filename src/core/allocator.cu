/**
 * @file allocator.cu
 * @brief CUDA memory allocator implementation
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 */

#ifdef PHOTON_USE_CUDA

#include "photon/core/allocator.hpp"

#include <cuda_runtime.h>

namespace photon {

Result<void*> CUDAAllocator::allocate(usize size, usize alignment) {
  if (size == 0) {
    return Err<void*>(ErrorCode::InvalidArgument,
                      "Cannot allocate zero bytes");
  }

  // Align size to alignment boundary
  size = (size + alignment - 1) & ~(alignment - 1);

  // Set device
  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    return Err<void*>(ErrorCode::CudaInvalidDevice,
                      std::string("cudaSetDevice failed: ") +
                          cudaGetErrorString(err));
  }

  // Allocate memory
  void* ptr = nullptr;
  err = cudaMalloc(&ptr, size);

  if (err != cudaSuccess) {
    if (err == cudaErrorMemoryAllocation) {
      return Err<void*>(ErrorCode::CudaOutOfMemory,
                        "CUDA out of memory: failed to allocate " +
                            std::to_string(size) + " bytes");
    }
    return Err<void*>(ErrorCode::CudaError,
                      std::string("cudaMalloc failed: ") +
                          cudaGetErrorString(err));
  }

  return Ok(ptr);
}

Result<void> CUDAAllocator::deallocate(void* ptr, [[maybe_unused]] usize size) {
  if (ptr == nullptr) {
    return Err<void>(ErrorCode::NullPointer,
                     "Cannot deallocate null pointer");
  }

  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                     std::string("cudaFree failed: ") +
                         cudaGetErrorString(err));
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
