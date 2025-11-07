/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file kv_cache_kernel.cu
 * @brief GPU kernels for efficient KV cache operations
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/kv_cache_kernel.cuh"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Batched KV cache write kernel
 *
 * Efficiently writes K and V tensors from a batch to their respective
 * cache positions in parallel, avoiding multiple cudaMemcpy calls.
 *
 * Grid: (batch_size, 1, 1) - one block per sequence
 * Block: (512, 1, 1) - 512 threads per block for parallel copying
 *
 * @param k_batch Source K tensor [batch_size, kv_dim]
 * @param v_batch Source V tensor [batch_size, kv_dim]
 * @param key_cache Destination key cache [total_capacity, kv_dim]
 * @param value_cache Destination value cache [total_capacity, kv_dim]
 * @param cache_offsets Cache starting offset for each sequence [batch_size]
 * @param positions Current position for each sequence [batch_size]
 * @param batch_size Number of sequences in the batch
 * @param kv_dim Dimension of K/V vectors
 */
__global__ void batched_kv_write_kernel(
    const float* __restrict__ k_batch,
    const float* __restrict__ v_batch,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    const i32* __restrict__ cache_offsets,
    const i32* __restrict__ positions,
    i32 batch_size,
    i32 kv_dim) {

  const int seq_idx = blockIdx.x;
  if (seq_idx >= batch_size) return;

  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // Calculate cache write position for this sequence
  const i32 cache_offset = cache_offsets[seq_idx];
  const i32 seq_pos = positions[seq_idx];
  const i32 cache_idx = cache_offset + seq_pos;

  // Source pointers (in k_batch/v_batch)
  const float* k_src = k_batch + seq_idx * kv_dim;
  const float* v_src = v_batch + seq_idx * kv_dim;

  // Destination pointers (in cache)
  float* k_dst = key_cache + cache_idx * kv_dim;
  float* v_dst = value_cache + cache_idx * kv_dim;

  // Parallel copy using grid-stride loop
  // Each thread copies multiple elements if kv_dim > num_threads
  for (int i = tid; i < kv_dim; i += num_threads) {
    k_dst[i] = k_src[i];
    v_dst[i] = v_src[i];
  }
}

Result<void> batched_kv_write_launch(
    const float* k_batch,
    const float* v_batch,
    float* key_cache,
    float* value_cache,
    const i32* cache_offsets,
    const i32* positions,
    i32 batch_size,
    i32 kv_dim,
    cudaStream_t stream) {

  if (!k_batch || !v_batch || !key_cache || !value_cache ||
      !cache_offsets || !positions) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Null pointer in batched_kv_write_launch");
  }

  if (batch_size <= 0 || kv_dim <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid dimensions for batched KV write");
  }

  // Launch configuration
  // - One block per sequence for parallel processing
  // - 512 threads per block (good for kv_dim=512 in Llama 3.2 1B)
  const int threads_per_block = 512;
  const int num_blocks = batch_size;

  batched_kv_write_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      k_batch, v_batch, key_cache, value_cache,
      cache_offsets, positions, batch_size, kv_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("batched_kv_write_kernel launch failed: ") +
                    cudaGetErrorString(err));
  }

  // No explicit sync - let the caller decide when to synchronize
  return Ok();
}

}  // namespace photon::kernels::cuda
