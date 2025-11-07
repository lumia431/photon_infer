/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file sampling_kernel.cu
 * @brief GPU kernels for token sampling (argmax, top-k, etc.)
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/sampling_kernel.cuh"
#include "photon/core/error.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief Optimized GPU kernel for batched argmax sampling
 *
 * Key optimizations:
 * 1. Coalesced memory access: threads access contiguous memory
 * 2. Block-strided loop for better cache utilization
 * 3. fmaxf() to reduce branch divergence
 * 4. Fewer warps (256 threads) for better occupancy
 *
 * @param logits Input logits [batch_size, vocab_size]
 * @param output Output token IDs [batch_size]
 * @param batch_size Number of sequences
 * @param vocab_size Vocabulary size
 */
__global__ void argmax_sampling_kernel(
    const float* __restrict__ logits,
    i32* __restrict__ output,
    i32 batch_size,
    i32 vocab_size) {

  const int seq_idx = blockIdx.x;
  if (seq_idx >= batch_size) return;

  const float* seq_logits = logits + seq_idx * vocab_size;

  // Shared memory for block-level reduction (max 32 warps for up to 1024 threads)
  __shared__ float shared_max_val[32];
  __shared__ i32 shared_max_idx[32];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_threads = blockDim.x;

  // Local max tracking
  float local_max = -INFINITY;
  i32 local_idx = 0;

  // Block-strided loop: each block processes CHUNK_SIZE elements at a time
  // This improves L1 cache hit rate vs strided access
  const int CHUNK_SIZE = 4096;  // Process 16KB chunks (better cache locality)

  for (int chunk_start = 0; chunk_start < vocab_size; chunk_start += CHUNK_SIZE) {
    int chunk_end = min(chunk_start + CHUNK_SIZE, vocab_size);

    // Vectorized loads within chunk (coalesced access)
    int start_idx = chunk_start + tid * 4;
    int end_idx = chunk_end;

    for (int i = start_idx; i < end_idx; i += num_threads * 4) {
      // Use float4 for vectorized loads when possible
      if (i + 3 < end_idx) {
        float4 vals = *reinterpret_cast<const float4*>(&seq_logits[i]);

        // Unroll comparison (fmaxf reduces branch divergence)
        float max_in_vec = fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w));
        if (max_in_vec > local_max) {
          // Find which element is max
          if (vals.x >= max_in_vec && vals.x > local_max) { local_max = vals.x; local_idx = i + 0; }
          else if (vals.y >= max_in_vec && vals.y > local_max) { local_max = vals.y; local_idx = i + 1; }
          else if (vals.z >= max_in_vec && vals.z > local_max) { local_max = vals.z; local_idx = i + 2; }
          else if (vals.w >= max_in_vec && vals.w > local_max) { local_max = vals.w; local_idx = i + 3; }
        }
      } else {
        // Handle remaining elements
        for (int j = i; j < end_idx && j < i + 4; ++j) {
          float val = seq_logits[j];
          if (val > local_max) {
            local_max = val;
            local_idx = j;
          }
        }
      }
    }
  }

  // Warp-level reduction using shuffle
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
    i32 other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
    if (other_val > local_max) {
      local_max = other_val;
      local_idx = other_idx;
    }
  }

  // First thread in each warp writes to shared memory
  if (lane_id == 0) {
    shared_max_val[warp_id] = local_max;
    shared_max_idx[warp_id] = local_idx;
  }

  __syncthreads();

  // Final reduction across warps (only first warp participates)
  if (warp_id == 0) {
    int num_warps = (num_threads + 31) / 32;
    float final_max = (tid < num_warps) ? shared_max_val[tid] : -INFINITY;
    i32 final_idx = (tid < num_warps) ? shared_max_idx[tid] : 0;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_val = __shfl_down_sync(0xffffffff, final_max, offset);
      i32 other_idx = __shfl_down_sync(0xffffffff, final_idx, offset);
      if (other_val > final_max) {
        final_max = other_val;
        final_idx = other_idx;
      }
    }

    // Thread 0 writes final result
    if (tid == 0) {
      output[seq_idx] = final_idx;
    }
  }
}

Result<void> argmax_sampling_launch(
    const float* logits,
    i32* output,
    i32 batch_size,
    i32 vocab_size,
    cudaStream_t stream) {

  if (!logits || !output) {
    return Err<void>(ErrorCode::InvalidArgument, "Null pointer in argmax_sampling_launch");
  }

  if (batch_size <= 0 || vocab_size <= 0) {
    return Err<void>(ErrorCode::InvalidArgument, "Invalid dimensions for argmax sampling");
  }

  // Use 512 threads as a balance between occupancy and parallelism
  // For 128K vocab: 512 threads process 256 elements each
  const int threads_per_block = 512;  // 16 warps
  const int num_blocks = batch_size;  // One block per sequence

  argmax_sampling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      logits, output, batch_size, vocab_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("argmax_sampling_kernel launch failed: ") +
                    cudaGetErrorString(err));
  }

  // NOTE: We don't synchronize here - let the caller decide when to sync.
  // The subsequent cudaMemcpy will implicitly synchronize anyway.
  return Ok();
}

}  // namespace photon::kernels::cuda
