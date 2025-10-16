/**
 * @file tensor.cpp
 * @brief Tensor class implementation
 */

#include "photon/core/tensor.hpp"

#include <numeric>
#include <sstream>

namespace photon {

// ============================================================================
// Helper Functions
// ============================================================================

usize Tensor::compute_size(const std::vector<int32_t>& dims) {
  if (dims.empty()) {
    return 1;  // Scalar
  }
  return std::accumulate(dims.begin(), dims.end(), usize(1),
                        [](usize a, int32_t b) { return a * b; });
}

// ============================================================================
// Constructors
// ============================================================================

Tensor::Tensor(Buffer&& buffer, std::vector<int32_t> dims, DataType dtype,
               DeviceType device)
    : buffer_(std::move(buffer)),
      dims_(std::move(dims)),
      size_(compute_size(dims_)),
      dtype_(dtype),
      device_(device) {}

Result<Tensor> Tensor::create(std::vector<int32_t> dims, DataType dtype,
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

Result<Tensor> Tensor::zeros(std::vector<int32_t> dims, DataType dtype,
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

Result<void> Tensor::reshape(const std::vector<int32_t>& new_dims) {
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

void Tensor::reset(DataType dtype, const std::vector<int32_t>& dims) {
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

}  // namespace photon
