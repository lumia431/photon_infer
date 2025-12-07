/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once


#include <photon/core/error.hpp>
#include <photon/core/types.hpp>

#include <span>
#include <cmath>
#include <algorithm>

#ifdef PHOTON_USE_EIGEN
#include <Eigen/Dense>
#endif

namespace photon::kernels {

/**
 * @brief Softmax kernel (naive implementation)
 *
 * Computes softmax with numerical stability:
 * 1. max_val = max(input)
 * 2. exp_sum = sum(exp(input - max_val))
 * 3. output[i] = exp(input[i] - max_val) / exp_sum
 *
 * @param input Input values [len]
 * @param output Output probabilities [len] (can be same as input for in-place)
 * @param len Number of elements
 */
template <FloatingPoint T>
void softmax_naive(std::span<const T> input, std::span<T> output, i32 len) noexcept {
  // Find max for numerical stability
  T max_val = input[0];
  for (i32 i = 1; i < len; ++i) {
    max_val = std::max(max_val, input[i]);
  }

  // Compute exp(x - max) and sum
  T sum = static_cast<T>(0);
  for (i32 i = 0; i < len; ++i) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  // Normalize
  T inv_sum = static_cast<T>(1) / sum;
  for (i32 i = 0; i < len; ++i) {
    output[i] *= inv_sum;
  }
}

#ifdef PHOTON_USE_EIGEN
/**
 * @brief Softmax kernel (Eigen implementation)
 *
 * Vectorized softmax using Eigen array operations
 */
template <FloatingPoint T>
Result<void> softmax_eigen(std::span<const T> input, std::span<T> output, i32 len) {
  using ArrayType = Eigen::Array<T, Eigen::Dynamic, 1>;
  using MapType = Eigen::Map<const ArrayType>;
  using MapTypeMut = Eigen::Map<ArrayType>;

  MapType input_arr(input.data(), len);
  MapTypeMut output_arr(output.data(), len);

  // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
  T max_val = input_arr.maxCoeff();
  output_arr = (input_arr - max_val).exp();
  T sum = output_arr.sum();
  output_arr /= sum;

  return Ok();
}
#endif

/**
 * @brief Dot product kernel (naive implementation)
 *
 * Computes: result = sum(a[i] * b[i] for i in 0..len-1)
 */
template <FloatingPoint T>
T dot_product_naive(std::span<const T> a, std::span<const T> b, i32 len) noexcept {
  T sum = static_cast<T>(0);
  for (i32 i = 0; i < len; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

#ifdef PHOTON_USE_EIGEN
/**
 * @brief Dot product kernel (Eigen implementation)
 */
template <FloatingPoint T>
T dot_product_eigen(std::span<const T> a, std::span<const T> b, i32 len) noexcept {
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MapType = Eigen::Map<const VectorType>;

  MapType a_vec(a.data(), len);
  MapType b_vec(b.data(), len);

  return a_vec.dot(b_vec);
}
#endif

/**
 * @brief Weighted sum accumulation kernel (naive)
 *
 * Computes: output += weight * value
 * where value is a vector and weight is a scalar
 *
 * @param value Input vector [len]
 * @param weight Scalar weight
 * @param output Accumulator vector [len] (modified in-place)
 * @param len Vector length
 */
template <FloatingPoint T>
void weighted_sum_naive(std::span<const T> value, T weight,
                       std::span<T> output, i32 len) noexcept {
  for (i32 i = 0; i < len; ++i) {
    output[i] += weight * value[i];
  }
}

#ifdef PHOTON_USE_EIGEN
/**
 * @brief Weighted sum accumulation kernel (Eigen)
 */
template <FloatingPoint T>
Result<void> weighted_sum_eigen(std::span<const T> value, T weight,
                               std::span<T> output, i32 len) {
  using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MapType = Eigen::Map<const VectorType>;
  using MapTypeMut = Eigen::Map<VectorType>;

  MapType value_vec(value.data(), len);
  MapTypeMut output_vec(output.data(), len);

  output_vec += weight * value_vec;

  return Ok();
}
#endif

/**
 * @brief Multi-Head Attention kernel (naive implementation)
 *
 * Computes full MHA for CPU:
 * 1. For each head h:
 *    a. Compute scores: score[h][t] = (Q[h] · K[h][t]) / sqrt(head_size)
 *    b. Apply softmax: attn[h] = softmax(score[h][0:pos+1])
 *    c. Aggregate values: output[h] = sum(attn[h][t] * V[h][t])
 *
 * @param query Query tensor [dim] where dim = head_num * head_size
 * @param key_cache Key cache [seq_len × kv_dim]
 * @param value_cache Value cache [seq_len × kv_dim]
 * @param output Output tensor [dim]
 * @param score Scratch buffer [head_num × seq_len] for attention scores
 * @param pos Current position (process positions 0 to pos)
 * @param dim Total query dimension
 * @param kv_dim Total key/value dimension
 * @param head_num Number of query heads
 * @param head_size Dimension per head
 * @param seq_len Maximum sequence length
 * @param kv_mul Query heads per KV head (for GQA)
 */
template <FloatingPoint T>
void mha_naive(
    std::span<const T> query,
    std::span<const T> key_cache,
    std::span<const T> value_cache,
    std::span<T> output,
    std::span<T> score,
    i32 pos,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 seq_len,
    i32 kv_mul) noexcept {

  const T scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(head_size));

  // Initialize output to zero
  std::fill(output.begin(), output.end(), static_cast<T>(0));

  // Process each head
  for (i32 h = 0; h < head_num; ++h) {
    const i32 q_offset = h * head_size;
    const i32 kv_head = h / kv_mul;  // Which KV head this query head uses
    const i32 kv_offset = kv_head * head_size;

    // Compute attention scores for all positions up to pos
    for (i32 t = 0; t <= pos; ++t) {
      const i32 key_base = t * kv_dim + kv_offset;

      // score[h][t] = dot(Q[h], K[t][kv_head]) * scale
      T dot = dot_product_naive<T>(
          query.subspan(q_offset, head_size),
          key_cache.subspan(key_base, head_size),
          head_size);

      score[h * seq_len + t] = dot * scale;
    }

    // Apply softmax to scores[h][0:pos+1]
    i32 score_len = pos + 1;
    softmax_naive<T>(
        score.subspan(h * seq_len, score_len),
        score.subspan(h * seq_len, score_len),
        score_len);

    // Weighted sum of values
    for (i32 t = 0; t <= pos; ++t) {
      const i32 value_base = t * kv_dim + kv_offset;
      T attn_weight = score[h * seq_len + t];

      weighted_sum_naive<T>(
          value_cache.subspan(value_base, head_size),
          attn_weight,
          output.subspan(q_offset, head_size),
          head_size);
    }
  }
}

#ifdef PHOTON_USE_EIGEN
/**
 * @brief Multi-Head Attention kernel (Eigen implementation)
 *
 * Optimized version using Eigen for vectorization
 */
template <FloatingPoint T>
Result<void> mha_eigen(
    std::span<const T> query,
    std::span<const T> key_cache,
    std::span<const T> value_cache,
    std::span<T> output,
    std::span<T> score,
    i32 pos,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 seq_len,
    i32 kv_mul) {

  const T scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(head_size));

  // Initialize output to zero
  std::fill(output.begin(), output.end(), static_cast<T>(0));

  // Process each head
  for (i32 h = 0; h < head_num; ++h) {
    const i32 q_offset = h * head_size;
    const i32 kv_head = h / kv_mul;
    const i32 kv_offset = kv_head * head_size;

    // Compute attention scores for all positions
    for (i32 t = 0; t <= pos; ++t) {
      const i32 key_base = t * kv_dim + kv_offset;

      T dot = dot_product_eigen<T>(
          query.subspan(q_offset, head_size),
          key_cache.subspan(key_base, head_size),
          head_size);

      score[h * seq_len + t] = dot * scale;
    }

    // Apply softmax
    i32 score_len = pos + 1;
    auto softmax_result = softmax_eigen<T>(
        score.subspan(h * seq_len, score_len),
        score.subspan(h * seq_len, score_len),
        score_len);

    if (!softmax_result) {
      return softmax_result;
    }

    // Weighted sum of values
    for (i32 t = 0; t <= pos; ++t) {
      const i32 value_base = t * kv_dim + kv_offset;
      T attn_weight = score[h * seq_len + t];

      auto ws_result = weighted_sum_eigen<T>(
          value_cache.subspan(value_base, head_size),
          attn_weight,
          output.subspan(q_offset, head_size),
          head_size);

      if (!ws_result) {
        return ws_result;
      }
    }
  }

  return Ok();
}
#else
/**
 * @brief Multi-Head Attention kernel (Eigen implementation disabled)
 *
 * This function is only available when PHOTON_USE_EIGEN is enabled.
 */
template <FloatingPoint T>
Result<void> mha_eigen(
    std::span<const T> /*query*/,
    std::span<const T> /*key_cache*/,
    std::span<const T> /*value_cache*/,
    std::span<T> /*output*/,
    std::span<T> /*score*/,
    i32 /*pos*/,
    i32 /*kv_dim*/,
    i32 /*head_num*/,
    i32 /*head_size*/,
    i32 /*seq_len*/,
    i32 /*kv_mul*/) {
  return Err<void>(ErrorCode::NotImplemented,
                  "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
}
#endif

}  // namespace photon::kernels

