/**
 * @file matmul_cublaslt.cuh
 * @brief cuBLASLt-based INT8 matrix multiplication with optimized performance
 * @version 1.0.0
 *
 * This implementation uses NVIDIA cuBLASLt library for high-performance INT8 GEMM,
 * significantly faster than custom kernels for medium to large matrices.
 *
 * Key features:
 * - INT8 GEMM with FP32 accumulation and output
 * - Support for both GEMV (vector) and GEMM (batched matrix)
 * - Automatic algorithm selection and heuristics
 * - Integrated dequantization (scale application)
 */

#pragma once

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <cublasLt.h>

namespace photon::kernels::cuda {

/**
 * @brief Initialize cuBLASLt handle (call once at startup)
 * @return Result containing handle or error
 */
Result<cublasLtHandle_t> cublaslt_init();

/**
 * @brief Cleanup cuBLASLt handle
 * @param handle cuBLASLt handle to destroy
 */
void cublaslt_cleanup(cublasLtHandle_t handle);

/**
 * @brief INT8 GEMV using cuBLASLt: output[K] = input[M] @ weight_int8[K×M]^T
 *
 * Computes matrix-vector multiplication with INT8 weights and FP32 input/output.
 * The weight matrix is quantized with per-group scales.
 *
 * Algorithm:
 * 1. Dequantize weight: weight_fp32[i] = weight_int8[i] * scales[group_idx]
 * 2. Compute GEMV: output = input @ weight_fp32^T
 *
 * Note: For small M/K, the custom kernel may be faster. Use this for M,K >= 512.
 *
 * @param handle cuBLASLt handle
 * @param input_ptr Input vector [M] (FP32)
 * @param input_size Size of input
 * @param weight_ptr Quantized weight matrix [K×M] (INT8, row-major)
 * @param weight_size Size of weight
 * @param scales_ptr Scale factors [num_groups] (FP32)
 * @param scales_size Size of scales
 * @param group_size Elements per quantization group
 * @param output_ptr Output vector [K] (FP32)
 * @param output_size Size of output
 * @param M Input dimension
 * @param K Output dimension
 * @param stream CUDA stream (nullptr for default)
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_cublaslt_launch(
    cublasLtHandle_t handle,
    const f32* input_ptr,
    usize input_size,
    const i8* weight_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 M,
    i32 K,
    cudaStream_t stream = nullptr);

/**
 * @brief Batched INT8 GEMM using cuBLASLt: output[B×K] = input[B×M] @ weight[K×M]^T
 *
 * Computes batched matrix multiplication with INT8 weights and FP32 input/output.
 *
 * This is the primary performance-critical operation in transformer inference.
 * cuBLASLt provides optimized implementations that can leverage:
 * - Tensor Core acceleration (on supported GPUs)
 * - Optimized memory access patterns
 * - Fused dequantization
 *
 * @param handle cuBLASLt handle
 * @param input_ptr Input matrix [B×M] (FP32)
 * @param input_size Size of input
 * @param weight_ptr Quantized weight matrix [K×M] (INT8, row-major)
 * @param weight_size Size of weight
 * @param scales_ptr Scale factors [num_groups] (FP32)
 * @param scales_size Size of scales
 * @param group_size Elements per quantization group
 * @param output_ptr Output matrix [B×K] (FP32)
 * @param output_size Size of output
 * @param B Batch size
 * @param K Output dimension
 * @param M Input dimension
 * @param stream CUDA stream (nullptr for default)
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_cublaslt_launch(
    cublasLtHandle_t handle,
    const f32* input_ptr,
    usize input_size,
    const i8* weight_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 B,
    i32 K,
    i32 M,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
