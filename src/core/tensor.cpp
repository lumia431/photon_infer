/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#include "photon/core/tensor.hpp"

#include <numeric>
#include <sstream>
#include <cstring>

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace photon {

// ============================================================================
// Helper Functions
// ============================================================================

usize Tensor::compute_size(const std::vector<i32>& dims) {
  if (dims.empty()) {
    return 1;  // Scalar
  }
  return std::accumulate(dims.begin(), dims.end(), usize(1),
                        [](usize a, i32 b) { return a * b; });
}

// ============================================================================
// Constructors
// ============================================================================

Tensor::Tensor(Buffer&& buffer, std::vector<i32> dims, DataType dtype,
               DeviceType device)
    : buffer_(std::move(buffer)),
      dims_(std::move(dims)),
      size_(compute_size(dims_)),
      dtype_(dtype),
      device_(device) {}

Result<Tensor> Tensor::create(std::vector<i32> dims, DataType dtype,
                              DeviceType device, bool need_alloc) {
  if (!need_alloc) {
    // Create empty tensor without allocation
    return Ok(Tensor(Buffer(), std::move(dims), dtype, device));
  }

  usize size = compute_size(dims);
  usize element_size = data_type_size(dtype);
  usize total_bytes = size * element_size;

  if (total_bytes == 0) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "Cannot create tensor with zero size");
  }

  // Create buffer for tensor data
  auto buffer_result = Buffer::create(total_bytes, device);
  if (!buffer_result) {
    return Err<Tensor>(std::move(buffer_result.error()));
  }

  return Ok(Tensor(std::move(buffer_result.value()), std::move(dims), dtype,
                  device));
}

Result<Tensor> Tensor::zeros(std::vector<i32> dims, DataType dtype,
                             DeviceType device) {
  auto result = create(dims, dtype, device);
  if (!result) {
    return result;
  }

  Tensor tensor = std::move(result.value());
  auto zero_result = tensor.buffer_.zero();
  if (!zero_result) {
    return Err<Tensor>(std::move(zero_result.error()));
  }

  return Ok(std::move(tensor));
}

// ============================================================================
// Getters
// ============================================================================

std::vector<usize> Tensor::strides() const {
  if (dims_.empty()) {
    return {};
  }

  std::vector<usize> strides(dims_.size());
  strides.back() = 1;

  // Compute strides in row-major order
  for (isize i = static_cast<isize>(dims_.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims_[i + 1];
  }

  return strides;
}

// ============================================================================
// Operations
// ============================================================================

Result<void> Tensor::reshape(const std::vector<i32>& new_dims) {
  usize new_size = compute_size(new_dims);

  if (new_size != size_) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Cannot reshape: element count mismatch. Current: " +
                        std::to_string(size_) +
                        ", New: " + std::to_string(new_size));
  }

  dims_ = new_dims;
  return Ok();
}

Result<Tensor> Tensor::clone() const {
  auto buffer_result = buffer_.clone();
  if (!buffer_result) {
    return Err<Tensor>(std::move(buffer_result.error()));
  }

  return Ok(Tensor(std::move(buffer_result.value()), dims_, dtype_, device_));
}

void Tensor::reset(DataType dtype, const std::vector<i32>& dims) {
  dtype_ = dtype;
  dims_ = dims;
  size_ = compute_size(dims);
  // Buffer is reset (becomes empty)
  buffer_ = Buffer();
}

Result<void> Tensor::to_cpu() {
  if (device_ == DeviceType::CPU) {
    return Ok();  // Already on CPU
  }

#ifdef PHOTON_USE_CUDA
  // TODO: Implement CUDA to CPU copy
  return Err<void>(ErrorCode::NotImplemented,
                  "CUDA to CPU copy not yet implemented");
#else
  return Err<void>(ErrorCode::InvalidArgument,
                  "CUDA not enabled in build");
#endif
}

Result<void> Tensor::to_cuda() {
  if (device_ == DeviceType::CUDA) {
    return Ok();  // Already on CUDA
  }

#ifdef PHOTON_USE_CUDA
  // TODO: Implement CPU to CUDA copy
  return Err<void>(ErrorCode::NotImplemented,
                  "CPU to CUDA copy not yet implemented");
#else
  return Err<void>(ErrorCode::InvalidArgument,
                  "CUDA not enabled in build");
#endif
}

std::string Tensor::to_string() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";

  for (size_t i = 0; i < dims_.size(); ++i) {
    oss << dims_[i];
    if (i < dims_.size() - 1) {
      oss << ", ";
    }
  }

  oss << "], dtype=" << data_type_str(dtype_)
      << ", device=" << device_type_str(device_) << ", size=" << size_
      << ")";

  return oss.str();
}

Result<Tensor> Tensor::to(DeviceType target_device) const {
  // If already on target device, return a copy
  if (device_ == target_device) {
    auto result = Tensor::create(dims_, dtype_, target_device);
    if (!result) {
      return result;
    }

    Tensor new_tensor = std::move(result.value());
    std::memcpy(new_tensor.data(), data(), byte_size());
    return Ok(std::move(new_tensor));
  }

#ifdef PHOTON_USE_CUDA
  // Device transfer between CPU and CUDA
  auto result = Tensor::create(dims_, dtype_, target_device);
  if (!result) {
    return result;
  }

  Tensor new_tensor = std::move(result.value());

  cudaMemcpyKind kind;
  if (device_ == DeviceType::CPU && target_device == DeviceType::CUDA) {
    kind = cudaMemcpyHostToDevice;
  } else if (device_ == DeviceType::CUDA && target_device == DeviceType::CPU) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    // CUDA to CUDA
    kind = cudaMemcpyDeviceToDevice;
  }

  cudaError_t err = cudaMemcpy(new_tensor.data(), data(), byte_size(), kind);
  if (err != cudaSuccess) {
    return Err<Tensor>(ErrorCode::CudaError,
                      std::string("Failed to copy tensor to device: ") +
                      cudaGetErrorString(err));
  }

  return Ok(std::move(new_tensor));
#else
  return Err<Tensor>(ErrorCode::NotImplemented,
                    "CUDA not available - cannot transfer to CUDA device");
#endif
}

// ============================================================================
// Tensor Slicing (Zero-Copy Views)
// ============================================================================

Tensor::Tensor(const Tensor& parent, std::vector<i32> dims, usize data_offset)
    : buffer_(),  // Empty buffer - we don't own the memory!
      dims_(std::move(dims)),
      size_(compute_size(dims_)),
      dtype_(parent.dtype_),
      device_(parent.device_) {
  // Calculate the actual data pointer
  // If parent is already a slice, use its data_ptr_, otherwise use buffer data
  void* base_ptr = parent.data_ptr_ ? parent.data_ptr_ : const_cast<void*>(parent.buffer_.data());
  data_ptr_ = static_cast<char*>(base_ptr) + data_offset;

  // Note: This creates a non-owning view. The parent tensor must outlive this slice!
}

Result<Tensor> Tensor::slice(i32 offset_rows, i32 num_rows) const {
  if (ndim() != 2) {
    return Err<Tensor>(ErrorCode::InvalidShape,
                      "slice() currently only supports 2D tensors");
  }

  i32 total_rows = dims_[0];
  i32 cols = dims_[1];

  if (offset_rows < 0 || offset_rows >= total_rows) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "offset_rows out of bounds: " + std::to_string(offset_rows));
  }

  if (num_rows <= 0 || offset_rows + num_rows > total_rows) {
    return Err<Tensor>(ErrorCode::InvalidArgument,
                      "Invalid num_rows: " + std::to_string(num_rows));
  }

  // Calculate byte offset
  usize element_size = data_type_size(dtype_);
  usize row_bytes = cols * element_size;
  usize byte_offset = offset_rows * row_bytes;

  // Create new dimensions for the slice
  std::vector<i32> slice_dims = {num_rows, cols};

  // Create the sliced tensor (shares memory!)
  Tensor sliced(*this, std::move(slice_dims), byte_offset);

  return Ok(std::move(sliced));
}

}  // namespace photon
