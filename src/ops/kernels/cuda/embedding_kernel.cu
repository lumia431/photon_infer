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

  // Each block handles one token
  const i32 token_index = blockIdx.x;
  
  // Bounds check for token index
  if (token_index < token_num) {
    // Retrieve token identifier and validate
    const i32 token_id = input_ptr[token_index];
    
    // Validate token ID
    if (token_id < vocab_size) {
      // Compute memory offsets
      const i32 output_offset = token_index * weight_dim;
      const i32 weight_offset = token_id * weight_dim;
      f32* output_base = output_ptr + output_offset;
      const f32* weight_base = weight_ptr + weight_offset;

      // Strided copy of embedding vector by each thread
      const i32 thread_id = threadIdx.x;
      const i32 block_size = blockDim.x;
      for (i32 dim_idx = thread_id; dim_idx < weight_dim; dim_idx += block_size) {
        output_base[dim_idx] = weight_base[dim_idx];
      }
    }
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
