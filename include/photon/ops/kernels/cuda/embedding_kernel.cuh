/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file embedding_kernel.cuh
 * @brief CUDA embedding kernel interface
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * h
 */


#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA embedding kernel
 *
 * Following standard design:
 * - Each block processes one token
 * - Each thread processes part of embedding dimension
 * - Grid size = number of tokens
 * - Block size = 128 threads
 *
 * @param tokens Input token IDs [num_tokens]
 * @param weight Embedding weight matrix [vocab_size × embedding_dim]
 * @param output Output embeddings [num_tokens × embedding_dim]
 * @param num_tokens Number of input tokens
 * @param vocab_size Vocabulary size
 * @param embedding_dim Embedding dimension
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> embedding_cuda_launch(
    std::span<const i32> tokens,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

