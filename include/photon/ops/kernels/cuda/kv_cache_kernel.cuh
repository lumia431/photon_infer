/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file kv_cache_kernel.cuh
 * @brief GPU kernels for efficient KV cache operations
 * @version 0.1.0
 */

#pragma once

#include "photon/core/types.hpp"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief Launch batched KV cache write kernel
 *
 * Efficiently writes K and V tensors from a batch to their respective
 * cache positions using a single GPU kernel, replacing multiple cudaMemcpy calls.
 *
 * @param k_batch Source K tensor [batch_size, kv_dim] (GPU)
 * @param v_batch Source V tensor [batch_size, kv_dim] (GPU)
 * @param key_cache Destination key cache [total_capacity, kv_dim] (GPU)
 * @param value_cache Destination value cache [total_capacity, kv_dim] (GPU)
 * @param cache_offsets Cache starting offset for each sequence [batch_size] (GPU)
 * @param positions Current position for each sequence [batch_size] (GPU)
 * @param batch_size Number of sequences in the batch
 * @param kv_dim Dimension of K/V vectors
 * @param stream CUDA stream for async execution (nullptr for default stream)
 * @return Result indicating success or error
 */
Result<void> batched_kv_write_launch(
    const float* k_batch,
    const float* v_batch,
    float* key_cache,
    float* value_cache,
    const i32* cache_offsets,
    const i32* positions,
    i32 batch_size,
    i32 kv_dim,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda
