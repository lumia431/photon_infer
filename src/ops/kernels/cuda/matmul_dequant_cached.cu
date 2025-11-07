/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file matmul_dequant_cached.cu
 * @brief Quantized GEMM with one-time dequantization and cuBLAS FP32 GEMM
 * @version 5.0.0
 *
 * Strategy (simple and effective):
 * 1. Dequantize INT8 weights to FP32 once per layer (cached)
 * 2. Use standard cuBLAS FP32 GEMM (fully optimized, Tensor Core for large matrices)
 * 3. Zero conversion overhead during inference
 *
 * Memory trade-off:
 * - INT8: ~250MB, FP32: ~1GB (acceptable on 8GB+ GPUs)
 * - Huge speedup from cuBLAS optimization
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Dequantize INT8 weights to FP32 with group-wise scaling (one-time operation)
 *
 * Grid: (K, (M + 255) / 256)
 * Block: 256 threads
 */
__global__ void dequantize_int8_to_fp32_kernel(
    const i8* __restrict__ weight_int8,  // [K × M]
    const f32* __restrict__ scales,      // [num_groups]
    f32* __restrict__ weight_fp32,       // [K × M]
    const i32 K,
    const i32 M,
    const i32 group_size) {

  const i32 row = blockIdx.x;
  const i32 col_base = blockIdx.y * blockDim.x;
  const i32 col = col_base + threadIdx.x;

  if (row >= K || col >= M) return;

  const i32 idx = row * M + col;
  const i8 val_int8 = weight_int8[idx];

  // Compute group index for this element
  const i32 group_idx = idx / group_size;
  const f32 scale = scales[group_idx];

  // Dequantize: FP32 = INT8 * scale
  weight_fp32[idx] = static_cast<f32>(val_int8) * scale;
}

/**
 * @brief Launch dequantization kernel
 */
Result<void> dequantize_int8_to_fp32_launch(
    const i8* weight_int8,
    const f32* scales,
    f32* weight_fp32,
    i32 K,
    i32 M,
    i32 group_size,
    cudaStream_t stream = nullptr) {

  // Kernel configuration
  constexpr int THREADS = 256;
  dim3 block(THREADS);
  dim3 grid(K, (M + THREADS - 1) / THREADS);

  // Launch kernel
  if (stream != nullptr) {
    dequantize_int8_to_fp32_kernel<<<grid, block, 0, stream>>>(
        weight_int8, scales, weight_fp32, K, M, group_size);
  } else {
    dequantize_int8_to_fp32_kernel<<<grid, block>>>(
        weight_int8, scales, weight_fp32, K, M, group_size);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    "CUDA kernel launch failed (dequant INT8→FP32): " +
                    std::string(cudaGetErrorString(err)));
  }

  return Ok();
}

/**
 * @brief Batched quantized GEMM with cached FP32 weights
 *
 * On first call: dequantizes INT8→FP32 and caches result
 * On subsequent calls: directly uses cached FP32 weights with cuBLAS
 */
Result<void> matmul_gemm_dequant_cached_launch(
    cublasHandle_t cublas_handle,
    const f32* input_ptr,
    usize input_size,
    const i8* weight_int8_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 batch_size,
    i32 K,
    i32 M,
    f32** weight_fp32_cache_ptr,
    cudaStream_t stream = nullptr) {

  // Validate dimensions
  if (input_size != static_cast<usize>(batch_size * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in dequant-cached GEMM");
  }

  if (weight_size != static_cast<usize>(K * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in dequant-cached GEMM");
  }

  if (output_size != static_cast<usize>(batch_size * K)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in dequant-cached GEMM");
  }

  // Set cuBLAS stream
  if (stream != nullptr) {
    cublasSetStream(cublas_handle, stream);
  }

  // ========================================================================
  // Step 1: Dequantize INT8 weights to FP32 (cached if available)
  // ========================================================================

  f32* weight_fp32 = nullptr;
  bool need_dequant = false;

  if (weight_fp32_cache_ptr == nullptr || *weight_fp32_cache_ptr == nullptr) {
    // Allocate FP32 weight buffer
    cudaError_t err = cudaMalloc(&weight_fp32, K * M * sizeof(f32));
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      "Failed to allocate FP32 weight buffer: " +
                      std::string(cudaGetErrorString(err)));
    }

    // Save to cache if pointer provided
    if (weight_fp32_cache_ptr != nullptr) {
      *weight_fp32_cache_ptr = weight_fp32;
    }

    need_dequant = true;
  } else {
    // Use cached FP32 weights
    weight_fp32 = *weight_fp32_cache_ptr;
    need_dequant = false;
  }

  // Dequantize if needed (only happens once per layer)
  if (need_dequant) {
    auto dequant_result = dequantize_int8_to_fp32_launch(
        weight_int8_ptr, scales_ptr, weight_fp32, K, M, group_size, stream);
    if (!dequant_result) {
      if (weight_fp32_cache_ptr == nullptr || *weight_fp32_cache_ptr == nullptr) {
        cudaFree(weight_fp32);
      }
      return dequant_result;
    }
  }

  // ========================================================================
  // Step 2: cuBLAS FP32 GEMM (standard, fully optimized)
  // ========================================================================
  // Compute: output[B×K] = input[B×M] @ weight[K×M]^T
  // In cuBLAS notation: C = alpha * A @ B^T + beta * C
  // where A = input[B×M], B = weight[K×M], C = output[B×K]

  const f32 alpha = 1.0f;
  const f32 beta = 0.0f;

  cublasStatus_t status = cublasSgemm(
      cublas_handle,
      CUBLAS_OP_T,    // Transpose B (weight)
      CUBLAS_OP_N,    // Don't transpose A (input)
      K,              // Rows of C
      batch_size,     // Columns of C
      M,              // Inner dimension
      &alpha,
      weight_fp32,    // B [K×M]
      M,              // Leading dimension of B
      input_ptr,      // A [B×M]
      M,              // Leading dimension of A
      &beta,
      output_ptr,     // C [B×K]
      K);             // Leading dimension of C

  if (status != CUBLAS_STATUS_SUCCESS) {
    if (weight_fp32_cache_ptr == nullptr || *weight_fp32_cache_ptr == nullptr) {
      cudaFree(weight_fp32);
    }
    return Err<void>(ErrorCode::CudaError,
                    "cuBLAS FP32 GEMM failed: " + std::to_string(status));
  }

  // Don't free cached weight
  return Ok();
}

}  // namespace photon::kernels::cuda
