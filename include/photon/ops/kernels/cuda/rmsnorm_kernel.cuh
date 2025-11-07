/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file rmsnorm_kernel.cuh
 * @brief CUDA RMS normalization kernel interface
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
 * @brief Launch CUDA RMS normalization kernel
 *
 * Following standard design:
 * - Computes: output = input * weight / sqrt(mean(input^2) + eps)
 * - Uses CUB BlockReduce for sum of squares
 * - Uses float4 vectorization
 * - Grid: 1 block, Block: 128 threads
 * - eps = 1e-5f
 *
 * @param input Input tensor [dim]
 * @param weight Weight tensor [dim]
 * @param output Output tensor [dim]
 * @param dim Dimension size
 * @param eps Epsilon for numerical stability (default 1e-5f)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps = 1e-5f,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch batched CUDA RMS normalization kernel
 *
 * Processes multiple sequences in parallel.
 * Grid: batch_size blocks, Block: 128 threads per block
 *
 * @param input Input tensor [batch_size × dim]
 * @param weight Weight tensor [dim] (shared across batch)
 * @param output Output tensor [batch_size × dim]
 * @param batch_size Number of sequences
 * @param dim Dimension size
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> rmsnorm_batched_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 batch_size,
    i32 dim,
    f32 eps = 1e-5f,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

