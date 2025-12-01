/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file matmul_kernel.cuh
 * @brief CUDA matrix multiplication kernel interface
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */


#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA matrix-vector multiplication (GEMV) kernel
 *
 * Following standard design:
 * - Computes: output[M] = input[N] @ weight[M×N]^T
 * - Each block computes 1 output element
 * - Uses float4 vectorization
 * - Uses CUB BlockReduce for efficient reduction
 * - Grid size = M, Block size = 128
 *
 * @param input Input vector [N]
 * @param weight Weight matrix [M × N]
 * @param output Output vector [M]
 * @param M Number of output elements (rows in weight)
 * @param N Input dimension (cols in weight)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 M,
    i32 N,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch batched GEMM using cuBLAS
 *
 * Computes: output[B×M] = input[B×N] @ weight[M×N]^T
 * Uses cuBLAS for optimal batched matrix multiplication.
 *
 * @param input Input matrix [B × N]
 * @param weight Weight matrix [M × N] (transposed during multiplication)
 * @param output Output matrix [B × M]
 * @param batch_size Number of sequences (B)
 * @param M Output dimension (rows in weight)
 * @param N Input dimension (cols in weight)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> matmul_gemm_cublas_launch(
    const f32* input,
    const f32* weight,
    f32* output,
    i32 batch_size,
    i32 M,
    i32 N,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

