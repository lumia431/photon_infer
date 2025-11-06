/**
 * @file matmul_kernel_quant_fp16.cu
 * @brief INT8 quantized GEMM using FP16 Tensor Cores - Implementation
 * @version 3.0.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"
#include "photon/ops/kernels/cuda/matmul_kernel_quant_fp16.cuh"

namespace photon::kernels::cuda {

// ============================================================================
// Dequantization Kernel: INT8 → FP16
// ============================================================================

/**
 * @brief Dequantize INT8 weights to FP16 with group-wise scaling
 *
 * Grid: (K, (M + 127) / 128)
 * Block: 128 threads
 *
 * Each thread processes one element, converting INT8 → FP16 with scale.
 */
__global__ void dequantize_int8_to_fp16_kernel(
    const i8* __restrict__ weight_int8,  // [K × M]
    const f32* __restrict__ scales,      // [num_groups]
    __half* __restrict__ weight_fp16,    // [K × M]
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

  // Dequantize: FP16 = INT8 * scale
  const f32 val_fp32 = static_cast<f32>(val_int8) * scale;
  weight_fp16[idx] = __float2half(val_fp32);
}

Result<void> dequantize_int8_to_fp16_launch(
    const i8* weight_int8,
    const f32* scales,
    __half* weight_fp16,
    i32 K,
    i32 M,
    i32 group_size,
    cudaStream_t stream) {

  // Kernel configuration
  constexpr int THREADS = 128;
  dim3 block(THREADS);
  dim3 grid(K, (M + THREADS - 1) / THREADS);

  // Launch kernel
  if (stream != nullptr) {
    dequantize_int8_to_fp16_kernel<<<grid, block, 0, stream>>>(
        weight_int8, scales, weight_fp16, K, M, group_size);
  } else {
    dequantize_int8_to_fp16_kernel<<<grid, block>>>(
        weight_int8, scales, weight_fp16, K, M, group_size);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    "CUDA kernel launch failed (dequant INT8→FP16): " +
                    std::string(cudaGetErrorString(err)));
  }

  return Ok();
}

// ============================================================================
// Type Conversion Kernels: FP32 ↔ FP16
// ============================================================================

__global__ void convert_fp32_to_fp16_kernel(
    const f32* __restrict__ input_fp32,
    __half* __restrict__ output_fp16,
    const i32 N) {

  const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  output_fp16[idx] = __float2half(input_fp32[idx]);
}

__global__ void convert_fp16_to_fp32_kernel(
    const __half* __restrict__ input_fp16,
    f32* __restrict__ output_fp32,
    const i32 N) {

  const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  output_fp32[idx] = __half2float(input_fp16[idx]);
}

// ============================================================================
// Main GEMM Function
// ============================================================================

Result<void> matmul_gemm_quant_fp16_launch(
    cublasHandle_t cublas_handle,
    const f32* input_ptr,
    usize input_size,
    const i8* weight_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 batch_size,
    i32 K,
    i32 M,
    __half** weight_fp16_cache_ptr,
    cudaStream_t stream) {

  // Validate dimensions
  if (input_size != static_cast<usize>(batch_size * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in FP16 quantized GEMM");
  }

  if (weight_size != static_cast<usize>(K * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in FP16 quantized GEMM");
  }

  if (output_size != static_cast<usize>(batch_size * K)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in FP16 quantized GEMM");
  }

  // Set cuBLAS stream
  if (stream != nullptr) {
    cublasSetStream(cublas_handle, stream);
  }

  // ========================================================================
  // Step 1: Dequantize INT8 weights to FP16 (cached if available)
  // ========================================================================

  __half* weight_fp16 = nullptr;
  bool need_dequant = false;

  if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
    // Allocate FP16 weight buffer
    cudaError_t err = cudaMalloc(&weight_fp16, K * M * sizeof(__half));
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      "Failed to allocate FP16 weight buffer: " +
                      std::string(cudaGetErrorString(err)));
    }

    // Save to cache if pointer provided
    if (weight_fp16_cache_ptr != nullptr) {
      *weight_fp16_cache_ptr = weight_fp16;
    }

    need_dequant = true;
  } else {
    // Use cached FP16 weights
    weight_fp16 = *weight_fp16_cache_ptr;
    need_dequant = false;
  }

  // Dequantize if needed
  if (need_dequant) {
    auto dequant_result = dequantize_int8_to_fp16_launch(
        weight_ptr, scales_ptr, weight_fp16, K, M, group_size, stream);
    if (!dequant_result) {
      if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
        cudaFree(weight_fp16);
      }
      return dequant_result;
    }
  }

  // ========================================================================
  // Step 2: Convert FP32 input to FP16
  // ========================================================================

  __half* input_fp16;
  cudaError_t err = cudaMalloc(&input_fp16, batch_size * M * sizeof(__half));
  if (err != cudaSuccess) {
    if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
      cudaFree(weight_fp16);
    }
    return Err<void>(ErrorCode::CudaError,
                    "Failed to allocate FP16 input buffer: " +
                    std::string(cudaGetErrorString(err)));
  }

  constexpr int THREADS = 256;
  i32 N = batch_size * M;
  i32 blocks = (N + THREADS - 1) / THREADS;

  if (stream != nullptr) {
    convert_fp32_to_fp16_kernel<<<blocks, THREADS, 0, stream>>>(
        input_ptr, input_fp16, N);
  } else {
    convert_fp32_to_fp16_kernel<<<blocks, THREADS>>>(
        input_ptr, input_fp16, N);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(input_fp16);
    if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
      cudaFree(weight_fp16);
    }
    return Err<void>(ErrorCode::CudaError,
                    "FP32→FP16 conversion failed: " +
                    std::string(cudaGetErrorString(err)));
  }

  // ========================================================================
  // Step 3: Allocate FP16 output buffer
  // ========================================================================

  __half* output_fp16;
  err = cudaMalloc(&output_fp16, batch_size * K * sizeof(__half));
  if (err != cudaSuccess) {
    cudaFree(input_fp16);
    if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
      cudaFree(weight_fp16);
    }
    return Err<void>(ErrorCode::CudaError,
                    "Failed to allocate FP16 output buffer: " +
                    std::string(cudaGetErrorString(err)));
  }

  // ========================================================================
  // Step 4: cuBLAS FP16 GEMM (Tensor Core accelerated)
  // ========================================================================
  // Compute: output[B×K] = input[B×M] @ weight[K×M]^T
  // In cuBLAS notation: C = alpha * A @ B^T + beta * C
  // where A = input[B×M], B = weight[K×M], C = output[B×K]

  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);

  cublasStatus_t status = cublasHgemm(
      cublas_handle,
      CUBLAS_OP_T,    // Transpose B (weight)
      CUBLAS_OP_N,    // Don't transpose A (input)
      K,              // Rows of C
      batch_size,     // Columns of C
      M,              // Inner dimension
      &alpha,
      weight_fp16,    // B [K×M]
      M,              // Leading dimension of B
      input_fp16,     // A [B×M]
      M,              // Leading dimension of A
      &beta,
      output_fp16,    // C [B×K]
      K);             // Leading dimension of C

  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(input_fp16);
    cudaFree(output_fp16);
    if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
      cudaFree(weight_fp16);
    }
    return Err<void>(ErrorCode::CudaError,
                    "cuBLAS FP16 GEMM failed: " + std::to_string(status));
  }

  // ========================================================================
  // Step 5: Convert FP16 output back to FP32
  // ========================================================================

  N = batch_size * K;
  blocks = (N + THREADS - 1) / THREADS;

  if (stream != nullptr) {
    convert_fp16_to_fp32_kernel<<<blocks, THREADS, 0, stream>>>(
        output_fp16, output_ptr, N);
  } else {
    convert_fp16_to_fp32_kernel<<<blocks, THREADS>>>(
        output_fp16, output_ptr, N);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(input_fp16);
    cudaFree(output_fp16);
    if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
      cudaFree(weight_fp16);
    }
    return Err<void>(ErrorCode::CudaError,
                    "FP16→FP32 conversion failed: " +
                    std::string(cudaGetErrorString(err)));
  }

  // ========================================================================
  // Cleanup (don't free cached weight)
  // ========================================================================

  cudaFree(input_fp16);
  cudaFree(output_fp16);

  // Only free weight if not cached
  if (weight_fp16_cache_ptr == nullptr || *weight_fp16_cache_ptr == nullptr) {
    cudaFree(weight_fp16);
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
