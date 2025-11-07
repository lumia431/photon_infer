/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file embedding_kernel.cu
 * @brief CUDA embedding kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/embedding_kernel.cuh"
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for embedding lookup
 *
 * Standard implementation:
 * - blockIdx.x = token index
 * - threadIdx.x = dimension index (strided)
 * - Grid: num_tokens blocks
 * - Block: 128 threads
 */
__global__ void emb_kernel_cu_fp32(
    i32 vocab_size,
    i32 token_num,
    i32 weight_dim,
    const i32* input_ptr,
    const f32* weight_ptr,
    f32* output_ptr) {

  // Following standard: each block processes one token
  i32 token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }

  // Get token ID and check validity
  i32 token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  // Calculate pointers (using standard approach exactly)
  f32* output_ptr_start = output_ptr + token_idx * weight_dim;
  const f32* weight_ptr_start = weight_ptr + token * weight_dim;

  // Each thread copies part of the embedding vector (using standard approach)
  for (i32 i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

Result<void> embedding_cuda_launch(
    std::span<const i32> tokens,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim,
    cudaStream_t stream) {

  // Validate input sizes
  if (static_cast<i32>(tokens.size()) != num_tokens) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Tokens size mismatch in embedding_cuda_launch");
  }

  if (static_cast<i32>(weight.size()) != vocab_size * embedding_dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in embedding_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != num_tokens * embedding_dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in embedding_cuda_launch");
  }

  // Launch configuration (using standard approach exactly)
  constexpr i32 thread_num = 128;

  // Launch kernel: grid size = num_tokens (one block per token)
  if (stream) {
    emb_kernel_cu_fp32<<<num_tokens, thread_num, 0, stream>>>(
        vocab_size, num_tokens, embedding_dim,
        tokens.data(), weight.data(), output.data());
  } else {
    emb_kernel_cu_fp32<<<num_tokens, thread_num>>>(
        vocab_size, num_tokens, embedding_dim,
        tokens.data(), weight.data(), output.data());
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA embedding kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA embedding kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
