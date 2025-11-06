/**
 * @file matmul_kernel_quant_fp16.cuh
 * @brief INT8 quantized GEMM using FP16 Tensor Cores
 * @version 3.0.0
 *
 * Strategy:
 * 1. Dequantize INT8 weights to FP16 on-the-fly (using group-wise scales)
 * 2. Convert FP32 input to FP16
 * 3. Use cuBLAS FP16 GEMM with Tensor Cores
 * 4. Convert FP16 output back to FP32
 *
 * Expected speedup: 2-3x over custom INT8 kernel (利用Tensor Core)
 */

#pragma once

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace photon::kernels::cuda {

/**
 * @brief Batched quantized GEMM using FP16 Tensor Cores
 *
 * This approach:
 * 1. Dequantizes INT8 weights to FP16 (once per layer, cached)
 * 2. Converts FP32 input to FP16
 * 3. Uses cuBLAS FP16 GEMM (Tensor Core accelerated)
 * 4. Converts FP16 output to FP32
 *
 * @param cublas_handle cuBLAS handle
 * @param input_ptr Input matrix [B × M] (FP32)
 * @param input_size Size of input
 * @param weight_ptr Quantized weight matrix [K × M] (INT8, row-major)
 * @param weight_size Size of weight
 * @param scales_ptr Scale factors [num_groups] (FP32)
 * @param scales_size Size of scales
 * @param group_size Elements per quantization group
 * @param output_ptr Output matrix [B × K] (FP32)
 * @param output_size Size of output
 * @param batch_size Batch size (B)
 * @param K Output dimension
 * @param M Input dimension
 * @param weight_fp16_cache_ptr Cached FP16 weights (nullptr to allocate)
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
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
    cudaStream_t stream = nullptr);

/**
 * @brief Dequantize INT8 weights to FP16 with group-wise scaling
 *
 * This kernel converts INT8 → FP16 applying group-wise scales.
 * Can be called once and cached for subsequent forward passes.
 *
 * @param weight_int8 INT8 weights [K × M]
 * @param scales Group-wise scales [num_groups]
 * @param weight_fp16 Output FP16 weights [K × M]
 * @param K Number of rows
 * @param M Number of columns
 * @param group_size Group size for quantization
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> dequantize_int8_to_fp16_launch(
    const i8* weight_int8,
    const f32* scales,
    __half* weight_fp16,
    i32 K,
    i32 M,
    i32 group_size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
