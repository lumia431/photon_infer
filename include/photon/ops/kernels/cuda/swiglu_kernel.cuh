/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file swiglu_kernel.cuh
 * @brief CUDA SwiGLU activation kernel interface
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
 * @brief Launch CUDA SwiGLU activation kernel
 *
 * Following standard design:
 * - SwiGLU: out = swish(in1) * in2
 * - Swish: swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
 * - Uses shared memory for caching inputs
 * - Grid: (size + 128 - 1) / 128 blocks
 * - Block: 128 threads
 * - Shared memory: 128 * sizeof(float) * 2
 *
 * @param input1 First input tensor (gate values)
 * @param input2 Second input tensor (values to modulate)
 * @param output Output tensor
 * @param size Number of elements
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch batched CUDA SwiGLU activation kernel
 *
 * Processes multiple sequences in parallel with a single kernel launch.
 *
 * @param input1 First input tensor [batch_size × hidden_dim] (gate values)
 * @param input2 Second input tensor [batch_size × hidden_dim] (values to modulate)
 * @param output Output tensor [batch_size × hidden_dim]
 * @param batch_size Number of sequences in batch
 * @param hidden_dim Hidden dimension size
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> swiglu_batched_cuda_launch(
    const f32* input1,
    const f32* input2,
    f32* output,
    i32 batch_size,
    i32 hidden_dim,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

