/**
 * @file rmsnorm_kernel.hpp
 * @brief CPU kernels for RMS normalization
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_KERNELS_RMSNORM_KERNEL_HPP
#define PHOTON_OPS_KERNELS_RMSNORM_KERNEL_HPP

#include <cmath>
#include <span>
#include <Eigen/Core>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels {

// ============================================================================
// Naive Implementation (Manual Loop)
// ============================================================================

/**
 * @brief Naive RMS normalization implementation
 *
 * Computes RMS normalization for a single vector:
 * ```
 * rms = sqrt(mean(input^2) + eps)
 * output = weight * (input / rms)
 * ```
 *
 * Algorithm:
 * 1. Compute sum of squares: sum = Σ(input[i]^2)
 * 2. Compute mean of squares: mean = sum / dim
 * 3. Compute RMS: rms = sqrt(mean + eps)
 * 4. Normalize and scale: output[i] = weight[i] * input[i] / rms
 *
 * Complexity: O(dim)
 *
 * @param input Input vector [dim]
 * @param weight Scaling factors [dim]
 * @param output Output vector [dim]
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <FloatingPoint T>
void rmsnorm_naive(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 dim,
    T eps) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(dim) ||
      weight.size() != static_cast<usize>(dim) ||
      output.size() != static_cast<usize>(dim)) {
    return;  // Silent failure in noexcept function
  }

  // Step 1: Compute sum of squares
  T sum_sq = 0;
  for (i32 i = 0; i < dim; ++i) {
    sum_sq += input[i] * input[i];
  }

  // Step 2: Compute mean of squares
  T mean_sq = sum_sq / static_cast<T>(dim);

  // Step 3: Compute RMS (rsqrt = 1 / rms for efficiency)
  T rms = std::sqrt(mean_sq + eps);
  T rsqrt = static_cast<T>(1.0) / rms;

  // Step 4: Normalize and scale
  for (i32 i = 0; i < dim; ++i) {
    output[i] = weight[i] * (input[i] * rsqrt);
  }
}

/**
 * @brief Naive batch RMS normalization
 *
 * Applies RMS normalization independently to each vector in a batch.
 *
 * Algorithm:
 * ```
 * for each batch item b:
 *   rms_b = sqrt(mean(input[b, :]^2) + eps)
 *   output[b, :] = weight * (input[b, :] / rms_b)
 * ```
 *
 * Complexity: O(batch_size × dim)
 *
 * @param input Input matrix [batch_size × dim] (row-major)
 * @param weight Scaling factors [dim]
 * @param output Output matrix [batch_size × dim] (row-major)
 * @param batch_size Number of vectors in batch
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <FloatingPoint T>
void rmsnorm_batch_naive(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 batch_size,
    i32 dim,
    T eps) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(batch_size * dim) ||
      weight.size() != static_cast<usize>(dim) ||
      output.size() != static_cast<usize>(batch_size * dim)) {
    return;
  }

  // Process each batch item independently
  for (i32 b = 0; b < batch_size; ++b) {
    // Offset for current batch item
    i32 offset = b * dim;

    // Step 1: Compute sum of squares for this batch item
    T sum_sq = 0;
    for (i32 i = 0; i < dim; ++i) {
      T val = input[offset + i];
      sum_sq += val * val;
    }

    // Step 2: Compute mean of squares
    T mean_sq = sum_sq / static_cast<T>(dim);

    // Step 3: Compute rsqrt
    T rms = std::sqrt(mean_sq + eps);
    T rsqrt = static_cast<T>(1.0) / rms;

    // Step 4: Normalize and scale
    for (i32 i = 0; i < dim; ++i) {
      output[offset + i] = weight[i] * (input[offset + i] * rsqrt);
    }
  }
}

// ============================================================================
// Optimized Implementation (Eigen)
// ============================================================================

/**
 * @brief Optimized RMS normalization using Eigen
 *
 * Uses Eigen's array operations for vectorization:
 * - SIMD vectorization (SSE/AVX)
 * - Efficient element-wise operations
 * - Auto-vectorized reduction (mean)
 *
 * Performance: ~2-3x faster than naive for typical dimensions
 *
 * @param input Input vector [dim]
 * @param weight Scaling factors [dim]
 * @param output Output vector [dim]
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 * @return Result indicating success or error
 */
template <FloatingPoint T>
Result<void> rmsnorm_eigen(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 dim,
    T eps) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_eigen");
  }
  if (weight.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_eigen");
  }
  if (output.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_eigen");
  }

  // Create Eigen array maps (zero-copy views)
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> input_arr(
      input.data(), dim);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> weight_arr(
      weight.data(), dim);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> output_arr(
      output.data(), dim);

  // Compute mean of squares: mean(input^2)
  T mean_sq = input_arr.square().mean();

  // Compute rsqrt: 1 / sqrt(mean_sq + eps)
  T rsqrt = static_cast<T>(1.0) / std::sqrt(mean_sq + eps);

  // Normalize and scale: output = weight * (input * rsqrt)
  output_arr = weight_arr * (input_arr * rsqrt);

  return Ok();
}

/**
 * @brief Optimized batch RMS normalization using Eigen
 *
 * Applies RMS normalization to each row of the batch matrix.
 *
 * Uses Eigen for:
 * - Vectorized row-wise operations
 * - Efficient memory access patterns
 * - SIMD optimizations
 *
 * Performance: ~2-3x faster than naive for typical batch sizes
 *
 * @param input Input matrix [batch_size × dim] (row-major)
 * @param weight Scaling factors [dim]
 * @param output Output matrix [batch_size × dim] (row-major)
 * @param batch_size Number of vectors in batch
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 * @return Result indicating success or error
 */
template <FloatingPoint T>
Result<void> rmsnorm_batch_eigen(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 batch_size,
    i32 dim,
    T eps) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(batch_size * dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_batch_eigen");
  }
  if (weight.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_batch_eigen");
  }
  if (output.size() != static_cast<usize>(batch_size * dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_batch_eigen");
  }

  // Create Eigen maps (zero-copy views)
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      input_mat(input.data(), batch_size, dim);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> weight_arr(
      weight.data(), dim);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      output_mat(output.data(), batch_size, dim);

  // Process each row (batch item)
  for (i32 b = 0; b < batch_size; ++b) {
    // Get row as array (zero-copy)
    auto input_row = input_mat.row(b).array();
    auto output_row = output_mat.row(b).array();

    // Compute mean of squares
    T mean_sq = input_row.square().mean();

    // Compute rsqrt
    T rsqrt = static_cast<T>(1.0) / std::sqrt(mean_sq + eps);

    // Normalize and scale (element-wise multiplication)
    output_row = weight_arr.transpose() * (input_row * rsqrt);
  }

  return Ok();
}

}  // namespace photon::kernels

#endif  // PHOTON_OPS_KERNELS_RMSNORM_KERNEL_HPP
