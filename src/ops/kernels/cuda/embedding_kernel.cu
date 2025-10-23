/**
 * @file embedding_kernel.cu
 * @brief CUDA implementation of embedding lookup
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/embedding_kernel.cuh"

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

template <i32 BLOCK_SIZE>
__global__ void embedding_kernel(
    const i32* __restrict__ tokens,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim) {

  // Each block processes one token
  const i32 token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  // Get token ID and validate
  const i32 token_id = tokens[token_idx];
  if (token_id < 0 || token_id >= vocab_size) {
    // Invalid token - fill with zeros
    for (i32 i = threadIdx.x; i < embedding_dim; i += BLOCK_SIZE) {
      output[token_idx * embedding_dim + i] = 0.0f;
    }
    return;
  }

  // Calculate base pointers
  const float* weight_base = weight + token_id * embedding_dim;
  float* output_base = output + token_idx * embedding_dim;

  // ============================================
  // Vectorized copy (float4)
  // ============================================
  constexpr i32 PACK_SIZE = 4;
  const i32 pack_num = embedding_dim / PACK_SIZE;
  const i32 pack_off = pack_num * PACK_SIZE;

  const i32 tid = threadIdx.x;

  // Vectorized part
  if (pack_num > 0) {
    const float4* weight_pack = reinterpret_cast<const float4*>(weight_base);
    float4* output_pack = reinterpret_cast<float4*>(output_base);

    for (i32 i = tid; i < pack_num; i += BLOCK_SIZE) {
      output_pack[i] = weight_pack[i];
    }
  }

  // Scalar tail
  for (i32 i = pack_off + tid; i < embedding_dim; i += BLOCK_SIZE) {
    output_base[i] = weight_base[i];
  }
}

// ============================================================================
// Host-side Launch Function
// ============================================================================

Result<void> embedding_cuda_launch(
    std::span<const i32> tokens,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 num_tokens,
    i32 vocab_size,
    i32 embedding_dim,
    cudaStream_t stream) {

  // Validate input sizes
  if (tokens.size() != static_cast<usize>(num_tokens)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Tokens size mismatch in embedding_cuda_launch");
  }

  if (weight.size() != static_cast<usize>(vocab_size * embedding_dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in embedding_cuda_launch");
  }

  if (output.size() != static_cast<usize>(num_tokens * embedding_dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in embedding_cuda_launch");
  }

  if (num_tokens <= 0 || vocab_size <= 0 || embedding_dim <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid dimensions in embedding_cuda_launch");
  }

  // Launch configuration
  constexpr i32 BLOCK_SIZE = EMBEDDING_BLOCK_SIZE;
  const i32 grid_size = num_tokens;

  if (grid_size > EMBEDDING_MAX_SEQ_LEN) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Number of tokens exceeds maximum sequence length");
  }

  if (stream != nullptr) {
    embedding_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        tokens.data(), weight.data(), output.data(),
        num_tokens, vocab_size, embedding_dim);
  } else {
    embedding_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        tokens.data(), weight.data(), output.data(),
        num_tokens, vocab_size, embedding_dim);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA embedding kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

template __global__ void embedding_kernel<EMBEDDING_BLOCK_SIZE>(
    const i32*, const float*, float*, i32, i32, i32);

}  // namespace photon::kernels::cuda
