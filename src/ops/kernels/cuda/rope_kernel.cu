/**
 * @file rope_kernel.cu
 * @brief CUDA implementation of Rotary Position Embedding
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/rope_kernel.cuh"

namespace photon::kernels::cuda {

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Apply 2D rotation to a pair of values (in-place)
 *
 * Computes:
 *   vec[idx]   = vec[idx] * cos - vec[idx+1] * sin
 *   vec[idx+1] = vec[idx] * sin + vec[idx+1] * cos
 *
 * (using float2 to read/write pairs atomically)
 */
__device__ inline void rope_rotate(
    float cos_val,
    float sin_val,
    float* vec,
    i32 idx) {

  // Read pair as float2
  float2* vec_pair = reinterpret_cast<float2*>(vec + idx);
  float2 val = *vec_pair;

  // Apply 2D rotation
  *vec_pair = make_float2(
      val.x * cos_val - val.y * sin_val,  // Real part
      val.x * sin_val + val.y * cos_val   // Imaginary part
  );
}

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

__global__ void compute_rope_cache_kernel(
    float* __restrict__ sin_cache,
    float* __restrict__ cos_cache,
    i32 max_seq_len,
    i32 head_size) {

  const i32 head_dim = threadIdx.x;
  if (head_dim >= head_size) {
    return;
  }

  // Compute frequency for this dimension
  // freq = 1.0 / (10000 ^ (head_dim / head_size))
  const float freq = 1.0f / powf(10000.0f,
                                 static_cast<float>(head_dim) /
                                 static_cast<float>(head_size));

  // Compute sin/cos for all positions
  for (i32 pos = 0; pos < max_seq_len; ++pos) {
    const float angle = static_cast<float>(pos) * freq;
    const i32 idx = pos * head_size + head_dim;

    sin_cache[idx] = sinf(angle);
    cos_cache[idx] = cosf(angle);
  }
}

template <i32 BLOCK_SIZE>
__global__ void rope_kernel(
    float* __restrict__ query,
    float* __restrict__ key,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size) {

  // Each thread processes pairs of elements (idx, idx+1)
  const i32 idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

  if (idx >= dim) {
    return;
  }

  // Get head dimension (for cache lookup)
  const i32 head_dim = idx % head_size;

  // Load sin/cos from cache
  const float sin_val = sin_cache[pos * head_size + head_dim];
  const float cos_val = cos_cache[pos * head_size + head_dim];

  // Always apply to Query
  rope_rotate(cos_val, sin_val, query, idx);

  // Apply to Key only if within kv_dim
  if (idx < kv_dim) {
    rope_rotate(cos_val, sin_val, key, idx);
  }
}

// ============================================================================
// Host-side Launch Functions
// ============================================================================

Result<void> compute_rope_cache_cuda_launch(
    std::span<f32> sin_cache,
    std::span<f32> cos_cache,
    i32 max_seq_len,
    i32 head_size,
    cudaStream_t stream) {

  // Validate sizes
  if (sin_cache.size() != static_cast<usize>(max_seq_len * head_size) ||
      cos_cache.size() != static_cast<usize>(max_seq_len * head_size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Cache size mismatch in compute_rope_cache_cuda_launch");
  }

  // Launch configuration: 1 block, head_size threads
  const i32 threads = head_size;

  if (stream != nullptr) {
    compute_rope_cache_kernel<<<1, threads, 0, stream>>>(
        sin_cache.data(), cos_cache.data(), max_seq_len, head_size);
  } else {
    compute_rope_cache_kernel<<<1, threads>>>(
        sin_cache.data(), cos_cache.data(), max_seq_len, head_size);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA compute_rope_cache kernel failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

Result<void> rope_cuda_launch(
    std::span<f32> query,
    std::span<f32> key,
    std::span<const f32> sin_cache,
    std::span<const f32> cos_cache,
    i32 pos,
    i32 dim,
    i32 kv_dim,
    i32 head_size,
    cudaStream_t stream) {

  // Validate inputs
  if (query.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Query size mismatch in rope_cuda_launch");
  }

  if (key.size() != static_cast<usize>(kv_dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Key size mismatch in rope_cuda_launch");
  }

  if (pos < 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid position in rope_cuda_launch");
  }

  // Launch configuration
  // Each thread processes 2 elements (a pair)
  constexpr i32 BLOCK_SIZE = 128;
  const i32 num_pairs = (dim + 1) / 2;  // Round up
  const i32 grid_size = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (stream != nullptr) {
    rope_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        query.data(), key.data(),
        sin_cache.data(), cos_cache.data(),
        pos, dim, kv_dim, head_size);
  } else {
    rope_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        query.data(), key.data(),
        sin_cache.data(), cos_cache.data(),
        pos, dim, kv_dim, head_size);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA rope kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

template __global__ void rope_kernel<128>(
    float*, float*, const float*, const float*, i32, i32, i32, i32);

}  // namespace photon::kernels::cuda
