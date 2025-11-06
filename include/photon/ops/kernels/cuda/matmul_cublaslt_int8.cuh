/**
 * @file matmul_cublaslt_int8.cuh
 * @brief cuBLASLt INT8 GEMM with Tensor Core support
 * @version 4.0.0
 *
 * Strategy:
 * 1. Convert group-wise scales to per-row scales (approximation)
 * 2. Use cuBLASLt INT8 GEMM with Tensor Core acceleration
 * 3. Apply per-row scaling in epilogue
 *
 * Note: This is an approximation since cuBLASLt doesn't support group-wise quantization.
 * We compute a representative scale for each output row.
 */

#pragma once

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <cublasLt.h>

namespace photon::kernels::cuda {

/**
 * @brief Batched quantized GEMM using cuBLASLt INT8 Tensor Core
 *
 * Approximates group-wise quantization with per-row scaling:
 * - Computes average scale for each weight row
 * - Uses cuBLASLt INT8 GEMM (Tensor Core accelerated)
 * - Applies per-row scaling after GEMM
 *
 * @param cublaslt_handle cuBLASLt handle
 * @param input_ptr Input matrix [B × M] (FP32)
 * @param input_size Size of input
 * @param weight_ptr Quantized weight matrix [K × M] (INT8, row-major)
 * @param weight_size Size of weight
 * @param scales_ptr Group-wise scale factors [num_groups] (FP32)
 * @param scales_size Size of scales
 * @param group_size Elements per quantization group
 * @param output_ptr Output matrix [B × K] (FP32)
 * @param output_size Size of output
 * @param batch_size Batch size (B)
 * @param K Output dimension
 * @param M Input dimension
 * @param per_row_scales_cache Cached per-row scales (nullptr to allocate)
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_cublaslt_int8_launch(
    cublasLtHandle_t cublaslt_handle,
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
    f32** per_row_scales_cache,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
