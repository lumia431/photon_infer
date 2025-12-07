/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file rope_kernel.cuh
 * @brief CUDA RoPE (Rotary Position Embedding) kernel interface
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
 * @brief Apply RoPE to Q and K tensors
 *
 * @param q Query tensor [dim]
 * @param k Key tensor [kv_dim]
 * @param sin_cache Sin cache [max_seq_len × head_size]
 * @param cos_cache Cos cache [max_seq_len × head_size]
 * @param pos Current position
 * @param dim Query dimension
 * @param kv_dim Key dimension
 * @param head_size Head size
 * @param stream CUDA stream (optional)
 */
Result<void> rope_cuda_launch(
    std::span<f32> q,
    std::span<f32> k,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

