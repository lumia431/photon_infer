/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file sampling_kernel.cuh
 * @brief GPU kernels for token sampling
 * @version 0.1.0
 */


#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief Launch argmax sampling kernel
 *
 * Finds the token with maximum logit for each sequence in the batch.
 * This is a GPU-based implementation that avoids copying logits to CPU.
 *
 * @param logits Input logits [batch_size, vocab_size] on GPU
 * @param output Output token IDs [batch_size] on GPU
 * @param batch_size Number of sequences
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream (nullptr for default stream)
 * @return Result<void> Success or error
 */
Result<void> argmax_sampling_launch(
    const float* logits,
    i32* output,
    i32 batch_size,
    i32 vocab_size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

