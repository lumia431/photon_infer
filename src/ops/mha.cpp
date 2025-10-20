#include "photon/ops/mha.hpp"
#include "photon/ops/kernels/mha_kernel.hpp"
#include <span>

namespace photon {

MHAOp::MHAOp(i32 dim, i32 kv_dim, i32 head_num, i32 head_size,
             i32 seq_len, bool use_naive)
    : dim_(dim),
      kv_dim_(kv_dim),
      head_num_(head_num),
      head_size_(head_size),
      seq_len_(seq_len),
      kv_mul_(head_num * head_size / kv_dim),
      use_naive_(use_naive) {
  device_ = DeviceType::CPU;
}

Result<void> MHAOp::forward(const Tensor& query, const Tensor& key_cache,
                           const Tensor& value_cache, Tensor& output, i32 pos) {
  // Validate inputs
  if (query.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Query tensor is empty");
  }
  if (key_cache.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Key cache tensor is empty");
  }
  if (value_cache.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Value cache tensor is empty");
  }
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  // Validate dimensions
  if (static_cast<i32>(query.size()) != dim_) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Query tensor size mismatch: expected " + std::to_string(dim_) +
                ", got " + std::to_string(query.size()));
  }

  if (static_cast<i32>(key_cache.size()) != seq_len_ * kv_dim_) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Key cache tensor size mismatch: expected " +
                std::to_string(seq_len_ * kv_dim_) +
                ", got " + std::to_string(key_cache.size()));
  }

  if (static_cast<i32>(value_cache.size()) != seq_len_ * kv_dim_) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Value cache tensor size mismatch: expected " +
                std::to_string(seq_len_ * kv_dim_) +
                ", got " + std::to_string(value_cache.size()));
  }

  if (static_cast<i32>(output.size()) != dim_) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Output tensor size mismatch: expected " + std::to_string(dim_) +
                ", got " + std::to_string(output.size()));
  }

  // Validate position
  if (pos < 0 || pos >= seq_len_) {
    return Err<void>(ErrorCode::InvalidArgument,
                "Position out of bounds: " + std::to_string(pos) +
                " (seq_len=" + std::to_string(seq_len_) + ")");
  }

  // Validate device compatibility
  if (query.device() != device_ || key_cache.device() != device_ ||
      value_cache.device() != device_ || output.device() != device_) {
    return Err<void>(ErrorCode::DeviceMismatch,
                "All tensors must be on the same device");
  }

  // Allocate scratch buffer for attention scores
  // Shape: [head_num × seq_len]
  auto score_result = Tensor::create({head_num_, seq_len_}, query.dtype(), device_);
  if (!score_result) {
    return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate score buffer");
  }
  Tensor score = std::move(score_result.value());

  // Dispatch to appropriate kernel based on dtype and device
  if (device_ == DeviceType::CPU) {
    if (query.dtype() == DataType::Float32) {
      // Get const maps for inputs
      auto query_map = query.vector_map<f32>();
      auto key_map = key_cache.vector_map<f32>();
      auto value_map = value_cache.vector_map<f32>();

      // Get mutable maps for outputs
      auto output_map = output.vector_map<f32>();
      auto score_map = score.vector_map<f32>();

      // Create spans from Eigen maps
      std::span<const f32> query_span(query_map.data(), query_map.size());
      std::span<const f32> key_span(key_map.data(), key_map.size());
      std::span<const f32> value_span(value_map.data(), value_map.size());
      std::span<f32> output_span(output_map.data(), output_map.size());
      std::span<f32> score_span(score_map.data(), score_map.size());

      if (use_naive_) {
        kernels::mha_naive<f32>(
            query_span, key_span, value_span,
            output_span, score_span,
            pos, kv_dim_, head_num_, head_size_, seq_len_, kv_mul_);
      } else {
        auto result = kernels::mha_eigen<f32>(
            query_span, key_span, value_span,
            output_span, score_span,
            pos, kv_dim_, head_num_, head_size_, seq_len_, kv_mul_);
        if (!result) {
          return result;
        }
      }
    } else if (query.dtype() == DataType::Float64) {
      auto query_map = query.vector_map<f64>();
      auto key_map = key_cache.vector_map<f64>();
      auto value_map = value_cache.vector_map<f64>();
      auto output_map = output.vector_map<f64>();
      auto score_map = score.vector_map<f64>();

      std::span<const f64> query_span(query_map.data(), query_map.size());
      std::span<const f64> key_span(key_map.data(), key_map.size());
      std::span<const f64> value_span(value_map.data(), value_map.size());
      std::span<f64> output_span(output_map.data(), output_map.size());
      std::span<f64> score_span(score_map.data(), score_map.size());

      if (use_naive_) {
        kernels::mha_naive<f64>(
            query_span, key_span, value_span,
            output_span, score_span,
            pos, kv_dim_, head_num_, head_size_, seq_len_, kv_mul_);
      } else {
        auto result = kernels::mha_eigen<f64>(
            query_span, key_span, value_span,
            output_span, score_span,
            pos, kv_dim_, head_num_, head_size_, seq_len_, kv_mul_);
        if (!result) {
          return result;
        }
      }
    } else {
      return Err<void>(ErrorCode::InvalidArgument,
                  "Unsupported data type for MHA operation");
    }
  } else {
    return Err<void>(ErrorCode::NotImplemented,
                "MHA operation not implemented for non-CPU devices");
  }

  return Ok();
}

}  // namespace photon
