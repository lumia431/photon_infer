/**
 * @file mha_kernel.cu
 * @brief CUDA implementation of Multi-Head Attention
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/mha_kernel.cuh"

#include <cub/cub.cuh>

namespace photon::kernels::cuda {

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief In-place softmax computation using CUB warp reduction
 *
 * @param x Input/output array (modified in-place)
 * @param size Number of elements to process
 */
__device__ void softmax_inplace(float* __restrict__ x, i32 size) {
  // Find max for numerical stability
  float max_val = -INFINITY;
  for (i32 i = 0; i < size; ++i) {
    max_val = fmaxf(max_val, x[i]);
  }

  // Compute exp(x - max) and sum
  float sum = 0.0f;
  for (i32 i = 0; i < size; ++i) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Normalize
  for (i32 i = 0; i < size; ++i) {
    x[i] /= sum;
  }
}

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

template <i32 BLOCK_SIZE>
__global__ void mha_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    float* __restrict__ score,
    i32 pos,
    i32 layer_idx,
    i32 seq_len,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 kv_mul) {

  // Each block processes one head
  const i32 head_idx = blockIdx.x;
  if (head_idx >= head_num) {
    return;
  }

  const i32 tid = threadIdx.x;
  const i32 kv_head_idx = head_idx / kv_mul;  // For GQA support

  // Compute pointers for this head
  const float* q_ptr = query + head_idx * head_size;
  const float* k_cache_ptr = key_cache + layer_idx * seq_len * kv_dim +
                             kv_head_idx * head_size;
  const float* v_cache_ptr = value_cache + layer_idx * seq_len * kv_dim +
                             kv_head_idx * head_size;
  float* out_ptr = output + head_idx * head_size;

  // Shared memory for attention scores and reduction
  extern __shared__ float shared_mem[];
  float* attn_scores = shared_mem;                    // [seq_len]
  float* temp_values = shared_mem + seq_len;          // [head_size]

  // ============================================
  // Phase 1: Compute attention scores Q·K^T
  // ============================================

  // Each thread computes dot product for one position
  for (i32 t = tid; t < pos + 1; t += BLOCK_SIZE) {
    // Q·K for position t
    float dot_product = 0.0f;

    // Vectorized computation (float4)
    constexpr i32 PACK_SIZE = 4;
    const i32 pack_num = head_size / PACK_SIZE;

    // Packed computation
    const float4* q_pack = reinterpret_cast<const float4*>(q_ptr);
    const float4* k_pack = reinterpret_cast<const float4*>(
        k_cache_ptr + t * kv_dim);

    for (i32 i = 0; i < pack_num; ++i) {
      float4 q_val = q_pack[i];
      float4 k_val = k_pack[i];

      dot_product += q_val.x * k_val.x + q_val.y * k_val.y +
                     q_val.z * k_val.z + q_val.w * k_val.w;
    }

    // Remainder
    for (i32 i = pack_num * PACK_SIZE; i < head_size; ++i) {
      dot_product += q_ptr[i] * k_cache_ptr[t * kv_dim + i];
    }

    // Scale by sqrt(head_size)
    attn_scores[t] = dot_product / sqrtf(static_cast<float>(head_size));
  }

  __syncthreads();

  // ============================================
  // Phase 2: Softmax normalization
  // ============================================

  if (tid == 0) {
    // Apply softmax to attention scores [0:pos]
    softmax_inplace(attn_scores, pos + 1);
  }

  __syncthreads();

  // ============================================
  // Phase 3: Weighted sum with V
  // ============================================

  // Initialize output for this head
  for (i32 i = tid; i < head_size; i += BLOCK_SIZE) {
    temp_values[i] = 0.0f;
  }

  __syncthreads();

  // Compute weighted sum: output = Σ attn_scores[t] * V[t]
  for (i32 t = 0; t < pos + 1; ++t) {
    const float attn_weight = attn_scores[t];

    // Vectorized accumulation
    constexpr i32 PACK_SIZE = 4;
    const i32 pack_num = head_size / PACK_SIZE;

    const float4* v_pack = reinterpret_cast<const float4*>(
        v_cache_ptr + t * kv_dim);

    // Packed computation
    for (i32 i = tid; i < pack_num; i += BLOCK_SIZE) {
      float4 v_val = v_pack[i];
      temp_values[i * PACK_SIZE + 0] += attn_weight * v_val.x;
      temp_values[i * PACK_SIZE + 1] += attn_weight * v_val.y;
      temp_values[i * PACK_SIZE + 2] += attn_weight * v_val.z;
      temp_values[i * PACK_SIZE + 3] += attn_weight * v_val.w;
    }

    // Remainder
    for (i32 i = pack_num * PACK_SIZE + tid; i < head_size; i += BLOCK_SIZE) {
      temp_values[i] += attn_weight * v_cache_ptr[t * kv_dim + i];
    }
  }

  __syncthreads();

  // Write final result
  for (i32 i = tid; i < head_size; i += BLOCK_SIZE) {
    out_ptr[i] = temp_values[i];
  }
}

// ============================================================================
// Host-side Launch Function
// ============================================================================

Result<void> mha_cuda_launch(
    std::span<const f32> query,
    std::span<const f32> key_cache,
    std::span<const f32> value_cache,
    std::span<f32> output,
    std::span<f32> score,
    i32 pos,
    i32 layer_idx,
    i32 seq_len,
    i32 kv_dim,
    i32 head_num,
    i32 head_size,
    i32 kv_mul,
    cudaStream_t stream) {

  // Validate inputs
  if (query.size() != static_cast<usize>(head_num * head_size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Query size mismatch in mha_cuda_launch");
  }

  if (key_cache.size() != static_cast<usize>(layer_idx + 1) * seq_len * kv_dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Key cache size mismatch in mha_cuda_launch");
  }

  if (value_cache.size() != static_cast<usize>(layer_idx + 1) * seq_len * kv_dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Value cache size mismatch in mha_cuda_launch");
  }

  if (output.size() != static_cast<usize>(head_num * head_size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in mha_cuda_launch");
  }

  if (score.size() != static_cast<usize>(head_num * seq_len)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Score buffer size mismatch in mha_cuda_launch");
  }

  if (pos < 0 || pos >= seq_len) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid position in mha_cuda_launch");
  }

  if (layer_idx < 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid layer index in mha_cuda_launch");
  }

  // Launch configuration
  constexpr i32 BLOCK_SIZE = 128;
  const i32 grid_size = head_num;  // One block per head
  const usize shmem_size = (seq_len + head_size) * sizeof(float);

  if (stream != nullptr) {
    mha_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, shmem_size, stream>>>(
        query.data(), key_cache.data(), value_cache.data(),
        output.data(), score.data(),
        pos, layer_idx, seq_len, kv_dim,
        head_num, head_size, kv_mul);
  } else {
    mha_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, shmem_size>>>(
        query.data(), key_cache.data(), value_cache.data(),
        output.data(), score.data(),
        pos, layer_idx, seq_len, kv_dim,
        head_num, head_size, kv_mul);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA MHA kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

template __global__ void mha_kernel<128>(
    const float*, const float*, const float*, float*, float*,
    i32, i32, i32, i32, i32, i32, i32);

}  // namespace photon::kernels::cuda
