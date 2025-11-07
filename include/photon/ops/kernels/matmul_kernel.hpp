/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file matmul_kernel.hpp
 * @brief CPU kernels for matrix multiplication
 * @version 0.1.0
 */


#include <span>
#include <Eigen/Core>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels {

// ============================================================================
// Naive Implementation (Triple Loop)
// ============================================================================

/**
 * @brief Naive matrix-vector multiplication: y = A @ x
 *
 * Computes: output[M] = weight[M × N] @ input[N]
 *
 * Algorithm:
 * ```
 * for i in 0..M:
 *   sum = 0
 *   for j in 0..N:
 *     sum += weight[i, j] * input[j]
 *   output[i] = sum
 * ```
 *
 * Complexity: O(M × N)
 * Cache: Poor spatial locality (row-major weight access)
 *
 * @param input Input vector [N]
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output vector [M]
 * @param M Output dimension
 * @param N Input dimension
 */
template <FloatingPoint T>
void matmul_gemv_naive(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 M,
    i32 N) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(N) ||
      weight.size() != static_cast<usize>(M * N) ||
      output.size() != static_cast<usize>(M)) {
    return;  // Silent failure in noexcept function
  }

  // Compute: output = weight @ input
  for (i32 i = 0; i < M; ++i) {
    T sum = 0;
    for (i32 j = 0; j < N; ++j) {
      sum += weight[i * N + j] * input[j];
    }
    output[i] = sum;
  }
}

/**
 * @brief Naive matrix-matrix multiplication: C = A @ B
 *
 * Computes: output[B × M] = input[B × N] @ weight[M × N]^T
 *
 * Algorithm:
 * ```
 * for b in 0..B:        # Batch
 *   for i in 0..M:      # Output rows
 *     sum = 0
 *     for j in 0..N:    # Reduction
 *       sum += input[b, j] * weight[i, j]
 *     output[b, i] = sum
 * ```
 *
 * Complexity: O(B × M × N)
 *
 * @param input Input matrix [B × N] (row-major)
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output matrix [B × M] (row-major)
 * @param B Batch size
 * @param M Output dimension
 * @param N Input dimension
 */
template <FloatingPoint T>
void matmul_gemm_naive(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 B,
    i32 M,
    i32 N) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(B * N) ||
      weight.size() != static_cast<usize>(M * N) ||
      output.size() != static_cast<usize>(B * M)) {
    return;
  }

  // Compute: output[b, i] = sum_j(input[b, j] * weight[i, j])
  for (i32 b = 0; b < B; ++b) {
    for (i32 i = 0; i < M; ++i) {
      T sum = 0;
      for (i32 j = 0; j < N; ++j) {
        sum += input[b * N + j] * weight[i * N + j];
      }
      output[b * M + i] = sum;
    }
  }
}

// ============================================================================
// Optimized Implementation (Eigen)
// ============================================================================

/**
 * @brief Optimized matrix-vector multiplication using Eigen
 *
 * Uses Eigen's optimized BLAS-like operations:
 * - SIMD vectorization (SSE/AVX)
 * - Loop unrolling
 * - Cache-friendly memory access
 *
 * Performance: ~10-100x faster than naive for large matrices
 *
 * @param input Input vector [N]
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output vector [M]
 * @param M Output dimension
 * @param N Input dimension
 * @return Result indicating success or error
 */
template <FloatingPoint T>
Result<void> matmul_gemv_eigen(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 M,
    i32 N) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in matmul_gemv");
  }
  if (weight.size() != static_cast<usize>(M * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in matmul_gemv");
  }
  if (output.size() != static_cast<usize>(M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in matmul_gemv");
  }

  // Create Eigen maps (zero-copy views)
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(
      input.data(), N);

  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      weight_mat(weight.data(), M, N);

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> output_vec(
      output.data(), M);

  // Compute: output = weight * input (optimized by Eigen)
  output_vec = weight_mat * input_vec;

  return Ok();
}

/**
 * @brief Optimized matrix-matrix multiplication using Eigen
 *
 * Computes: output[B × M] = input[B × N] @ weight[M × N]^T
 *
 * Uses Eigen's optimized matrix multiplication:
 * - Block-based algorithm for cache efficiency
 * - Multi-threading (if enabled)
 * - SIMD vectorization
 *
 * Performance: ~100-1000x faster than naive for large matrices
 *
 * @param input Input matrix [B × N] (row-major)
 * @param weight Weight matrix [M × N] (row-major)
 * @param output Output matrix [B × M] (row-major)
 * @param B Batch size
 * @param M Output dimension
 * @param N Input dimension
 * @return Result indicating success or error
 */
template <FloatingPoint T>
Result<void> matmul_gemm_eigen(
    std::span<const T> input,
    std::span<const T> weight,
    std::span<T> output,
    i32 B,
    i32 M,
    i32 N) noexcept {

  // Validate sizes
  if (input.size() != static_cast<usize>(B * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in matmul_gemm");
  }
  if (weight.size() != static_cast<usize>(M * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in matmul_gemm");
  }
  if (output.size() != static_cast<usize>(B * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in matmul_gemm");
  }

  // Create Eigen maps (zero-copy views)
  // input: [B × N]
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      input_mat(input.data(), B, N);

  // weight: [M × N]
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      weight_mat(weight.data(), M, N);

  // output: [B × M]
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      output_mat(output.data(), B, M);

  // Compute: output = input * weight^T
  // input[B×N] @ weight[M×N]^T = input[B×N] @ [N×M] = output[B×M]
  output_mat = input_mat * weight_mat.transpose();

  return Ok();
}

}  // namespace photon::kernels

