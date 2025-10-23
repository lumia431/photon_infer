/**
 * @file rmsnorm_kernel.cu
 * @brief CUDA implementation of RMS normalization kernels
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/rmsnorm_kernel.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Implementation
// ============================================================================

template <i32 BLOCK_DIM>
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 dim,
    float eps) {

  // Extremely simple kernel for debugging: just copy input to output
  const i32 tid = threadIdx.x;

  // Each thread copies one element if within bounds
  if (tid < dim) {
    output[tid] = input[tid];  // Simple copy for now
  }
}

template <i32 BLOCK_DIM>
__global__ void rmsnorm_batch_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 batch_size,
    i32 dim,
    float eps) {

  const i32 tid = threadIdx.x;
  const i32 batch_idx = blockIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  // Offset for current batch item
  const i32 offset = batch_idx * dim;
  const float* batch_input = input + offset;
  float* batch_output = output + offset;

  // Vectorization configuration
  constexpr i32 PACK_SIZE = 4;
  const i32 pack_num = dim / PACK_SIZE;
  const i32 pack_off = pack_num * PACK_SIZE;

  // ============================================
  // Phase 1: Compute sum of squares
  // ============================================
  float sum_sq = 0.0f;

  const float4* input_pack = reinterpret_cast<const float4*>(batch_input);
  for (i32 i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 val = input_pack[i];
    sum_sq += val.x * val.x;
    sum_sq += val.y * val.y;
    sum_sq += val.z * val.z;
    sum_sq += val.w * val.w;
  }

  for (i32 i = pack_off + tid; i < dim; i += BLOCK_DIM) {
    float val = batch_input[i];
    sum_sq += val * val;
  }

  // ============================================
  // Phase 2: Block-level reduction
  // ============================================
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float shared_sum;

  sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

  if (tid == 0) {
    shared_sum = sum_sq;
  }
  __syncthreads();

  sum_sq = shared_sum;

  // ============================================
  // Phase 3: Compute scale factor
  // ============================================
  const float mean_sq = sum_sq / static_cast<float>(dim);
  const float rsqrt = rsqrtf(mean_sq + eps);

  // ============================================
  // Phase 4: Normalize and scale
  // ============================================
  const float4* weight_pack = reinterpret_cast<const float4*>(weight);
  float4* output_pack = reinterpret_cast<float4*>(batch_output);

  for (i32 i = tid; i < pack_num; i += BLOCK_DIM) {
    float4 in_val = input_pack[i];
    float4 w_val = weight_pack[i];
    output_pack[i] = make_float4(
        rsqrt * in_val.x * w_val.x,
        rsqrt * in_val.y * w_val.y,
        rsqrt * in_val.z * w_val.z,
        rsqrt * in_val.w * w_val.w);
  }

  for (i32 i = pack_off + tid; i < dim; i += BLOCK_DIM) {
    batch_output[i] = rsqrt * batch_input[i] * weight[i];
  }
}

// ============================================================================
// Host-side Launch Functions
// ============================================================================

Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps,
    cudaStream_t stream) {

  // Validate input sizes
  if (dim <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid dimension in rmsnorm_cuda_launch");
  }

  if (input.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_cuda_launch");
  }

  if (weight.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_cuda_launch");
  }

  if (output.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_cuda_launch");
  }

  // Check alignment for vectorization
  if (dim % RMSNORM_PACK_SIZE != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Dimension must be multiple of 4 for vectorization");
  }

  // Launch kernel
  constexpr i32 BLOCK_SIZE = RMSNORM_BLOCK_SIZE;
  const i32 grid_size = 1;

  if (stream != nullptr) {
    rmsnorm_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input.data(), weight.data(), output.data(), dim, eps);
  } else {
    rmsnorm_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        input.data(), weight.data(), output.data(), dim, eps);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

Result<void> rmsnorm_batch_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 batch_size,
    i32 dim,
    f32 eps,
    cudaStream_t stream) {

  // Validate input sizes
  if (batch_size <= 0 || dim <= 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Invalid batch_size or dim in rmsnorm_batch_cuda_launch");
  }

  if (input.size() != static_cast<usize>(batch_size * dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_batch_cuda_launch");
  }

  if (weight.size() != static_cast<usize>(dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_batch_cuda_launch");
  }

  if (output.size() != static_cast<usize>(batch_size * dim)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_batch_cuda_launch");
  }

  if (dim % RMSNORM_PACK_SIZE != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Dimension must be multiple of 4 for vectorization");
  }

  // Launch kernel: one block per batch item
  constexpr i32 BLOCK_SIZE = RMSNORM_BLOCK_SIZE;
  const i32 grid_size = batch_size;

  if (stream != nullptr) {
    rmsnorm_batch_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input.data(), weight.data(), output.data(), batch_size, dim, eps);
  } else {
    rmsnorm_batch_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        input.data(), weight.data(), output.data(), batch_size, dim, eps);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

// Explicitly instantiate for common block sizes
template __global__ void rmsnorm_kernel<128>(
    const float*, const float*, float*, i32, float);
template __global__ void rmsnorm_batch_kernel<128>(
    const float*, const float*, float*, i32, i32, float);

}  // namespace photon::kernels::cuda
