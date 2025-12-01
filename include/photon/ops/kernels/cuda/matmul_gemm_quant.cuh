/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file matmul_gemm_quant.cuh
 * @brief Optimized INT8 quantized GEMM kernel - header
 * @version 2.0.0
 */


#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief Optimized batched quantized GEMM: output[B×K] = input[B×M] @ weight[K×M]^T
 *
 * Optimizations:
 * - Vectorized memory access (4x INT8 weights per load)
 * - Shared memory caching for input vectors
 * - Optimized thread block configuration (256 threads)
 * - Reduced global memory accesses
 *
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
 * @param K Output dimension (number of output features)
 * @param M Input dimension (number of input features)
 * @param stream CUDA stream (nullptr for default stream)
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_quant_launch(
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
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
