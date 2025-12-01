/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file matmul_dequant_cached.cuh
 * @brief Quantized GEMM with one-time dequantization header
 * @version 5.0.0
 */

#pragma once

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace photon::kernels::cuda {

/**
 * @brief Batched quantized GEMM with cached FP32 dequantization
 *
 * Strategy:
 * - First call: Dequantize INT8→FP32 and cache (one-time cost)
 * - Subsequent calls: Use cached FP32 weights with cuBLAS SGEMM
 * - Zero conversion overhead, full cuBLAS optimization
 *
 * Expected performance: 3-5x faster than custom INT8 kernel
 *
 * @param cublas_handle cuBLAS handle
 * @param input_ptr Input matrix [B × M] (FP32)
 * @param input_size Size of input
 * @param weight_int8_ptr Quantized weight matrix [K × M] (INT8)
 * @param weight_size Size of weight
 * @param scales_ptr Group-wise scale factors (FP32)
 * @param scales_size Size of scales
 * @param group_size Group size for quantization
 * @param output_ptr Output matrix [B × K] (FP32)
 * @param output_size Size of output
 * @param batch_size Batch size (B)
 * @param K Output dimension
 * @param M Input dimension
 * @param weight_fp32_cache_ptr Cached FP32 weights (nullptr to allocate)
 * @param stream CUDA stream
 * @return Result indicating success or error
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
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
