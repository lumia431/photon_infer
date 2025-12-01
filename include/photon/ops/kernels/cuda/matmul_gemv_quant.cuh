/**
 * @file matmul_gemv_quant.cuh
 * @brief CUDA kernel declarations for quantized matrix-vector multiplication (GEMV)
 * @version 0.1.0
 */
#pragma once


#include <cuda_runtime.h>
#include <span>

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch quantized GEMV kernel
 *
 * Performs: output[K] = input[M] @ weight_int8[K × M]^T
 * with dynamic dequantization using per-group scales.
 *
 * @param input_ptr Input activations [M] (float32)
 * @param input_size Size of input
 * @param weight_ptr Quantized weights [K × M] (int8)
 * @param weight_size Size of weight
 * @param scales_ptr Per-group scale factors (float32)
 * @param scales_size Size of scales
 * @param group_size Number of elements per quantization group
 * @param output_ptr Output activations [K] (float32)
 * @param output_size Size of output
 * @param M Input dimension
 * @param K Output dimension
 * @param stream CUDA stream (nullptr for default)
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_quant_launch(
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
 * @brief Launch batched quantized GEMM
 *
 * Computes: output[B×K] = input[B×M] @ weight_int8[K×M]^T
 * Uses warp-level parallelism for efficient batched computation.
 *
 * @param input Input matrix [B × M] (float32)
 * @param weight_quant Quantized weights [K × M] (int8)
 * @param scales Per-group scale factors
 * @param group_size Quantization group size
 * @param output Output matrix [B × K] (float32)
 * @param batch_size Number of sequences (B)
 * @param M Input dimension
 * @param K Output dimension
 * @param stream CUDA stream
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_quant_batched_launch(
    const f32* input,
    const i8* weight_quant,
    const f32* scales,
    i32 group_size,
    f32* output,
    i32 batch_size,
    i32 M,
    i32 K,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

