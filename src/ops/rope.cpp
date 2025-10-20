/**
 * @file rope.cpp
 * @brief Rotary Position Embedding operator implementation
 * @version 0.1.0
 */

#include "photon/ops/rope.hpp"
#include "photon/ops/kernels/rope_kernel.hpp"

namespace photon {

// ============================================================================
// RoPEOp Implementation
// ============================================================================

Result<void> RoPEOp::init_impl() {
  // Allocate sin/cos cache: [max_seq_len × head_size]
  usize cache_size = static_cast<usize>(max_seq_len_) * static_cast<usize>(head_size_);
  sin_cache_.resize(cache_size);
  cos_cache_.resize(cache_size);

  // Precompute sin/cos values for all positions
  kernels::compute_rope_cache<f32>(
      std::span<f32>(sin_cache_),
      std::span<f32>(cos_cache_),
      max_seq_len_,
      head_size_);

  return Ok();
}

Result<void> RoPEOp::forward(Tensor& q, Tensor& k, i32 pos) {
  // Check initialization
  if (!is_initialized()) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "RoPE operator not initialized");
  }

  // Validate position
  if (pos < 0 || pos >= max_seq_len_) {
    return Err<void>(
        ErrorCode::InvalidArgument,
        "Position out of range: expected [0, " + std::to_string(max_seq_len_) +
            "), got " + std::to_string(pos));
  }

  // Validate query tensor
  if (q.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Query tensor is empty");
  }

  if (q.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Query must be Float32");
  }

  if (q.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape, "Query must be 1D tensor");
  }

  if (q.dim(0) != dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Query dimension mismatch: expected " + std::to_string(dim_) +
            ", got " + std::to_string(q.dim(0)));
  }

  // Validate key tensor
  if (k.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Key tensor is empty");
  }

  if (k.dtype() != DataType::Float32) {
    return Err<void>(ErrorCode::InvalidDtype, "Key must be Float32");
  }

  if (k.ndim() != 1) {
    return Err<void>(ErrorCode::InvalidShape, "Key must be 1D tensor");
  }

  if (k.dim(0) != kv_dim_) {
    return Err<void>(
        ErrorCode::ShapeMismatch,
        "Key dimension mismatch: expected " + std::to_string(kv_dim_) +
            ", got " + std::to_string(k.dim(0)));
  }

  // Dispatch to device-specific implementation
  if (device_ == DeviceType::CPU && q.device() == DeviceType::CPU &&
      k.device() == DeviceType::CPU) {
    return forward_cpu(q, k, pos);
  }

#ifdef PHOTON_USE_CUDA
  if (device_ == DeviceType::CUDA && q.device() == DeviceType::CUDA &&
      k.device() == DeviceType::CUDA) {
    return forward_cuda(q, k, pos);
  }
#endif

  return Err<void>(ErrorCode::DeviceMismatch,
                  "Tensor device mismatch with operator device");
}

Result<void> RoPEOp::forward_cpu(Tensor& q, Tensor& k, i32 pos) {
  // Create spans for kernels
  std::span<f32> q_data(q.ptr<f32>(), q.size());
  std::span<f32> k_data(k.ptr<f32>(), k.size());
  std::span<const f32> sin_data(sin_cache_.data(), sin_cache_.size());
  std::span<const f32> cos_data(cos_cache_.data(), cos_cache_.size());

  // Dispatch based on implementation type
  if (use_naive_) {
    kernels::rope_naive<f32>(
        q_data, k_data, sin_data, cos_data,
        pos, dim_, kv_dim_, head_size_);
    return Ok();
  } else {
    return kernels::rope_eigen<f32>(
        q_data, k_data, sin_data, cos_data,
        pos, dim_, kv_dim_, head_size_);
  }
}

#ifdef PHOTON_USE_CUDA
Result<void> RoPEOp::forward_cuda(Tensor& q, Tensor& k, i32 pos) {
  return Err<void>(ErrorCode::NotImplemented,
                  "CUDA RoPE not yet implemented");
}
#endif

}  // namespace photon
