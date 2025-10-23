/**
 * @file matmul_kernel.cu
 * @brief CUDA implementation of matrix multiplication kernels
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/matmul_kernel.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel Implementation - GEMV (Matrix-Vector)
// ============================================================================

template <i32 THREAD_PER_BLOCK, i32 ROW_PER_BLOCK>
__global__ void matmul_gemv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 M,
    i32 N) {

  __shared__ float sdata[THREAD_PER_BLOCK];
  const i32 tid = threadIdx.x;

  // Each block processes ROW_PER_BLOCK rows
  const i32 start_row = blockIdx.x * ROW_PER_BLOCK;
  const i32 end_row = min(start_row + ROW_PER_BLOCK, M);

  if (start_row >= M) {
    return;
  }

  // Vectorization configuration
  constexpr i32 PACK_SIZE = 4;
  const i32 pack_num = N / PACK_SIZE;
  const i32 pack_off = pack_num * PACK_SIZE;

  const float4* input_pack = reinterpret_cast<const float4*>(input);

  // Process each row assigned to this block
  #pragma unroll
  for (i32 row = start_row; row < end_row; ++row) {
    // Initialize thread-local accumulator
    sdata[tid] = 0.0f;

    const i32 row_offset = row * N;
    const float4* weight_pack = reinterpret_cast<const float4*>(weight + row_offset);

    // ============================================
    // Phase 1: Vectorized dot product
    // ============================================
    #pragma unroll
    for (i32 i = tid; i < pack_num; i += THREAD_PER_BLOCK) {
      float4 in_val = input_pack[i];
      float4 w_val = weight_pack[i];

      // Dot product of 4-element vectors
      float partial_sum = in_val.x * w_val.x +
                         in_val.y * w_val.y +
                         in_val.z * w_val.z +
                         in_val.w * w_val.w;
      sdata[tid] += partial_sum;
    }

    // ============================================
    // Phase 2: Scalar tail
    // ============================================
    for (i32 i = pack_off + tid; i < N; i += THREAD_PER_BLOCK) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // ============================================
    // Phase 3: Block-level reduction
    // ============================================
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sdata[tid]);
    __syncthreads();

    // Thread 0 writes final result
    if (tid == 0) {
      output[row] = block_sum;
    }
    __syncthreads();
  }
}

// ============================================================================
// CUDA Kernel Implementation - GEMM (Matrix-Matrix)
// ============================================================================

template <i32 THREAD_PER_BLOCK, i32 ROW_PER_BLOCK>
__global__ void matmul_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    i32 B,
    i32 M,
    i32 N) {

  __shared__ float sdata[THREAD_PER_BLOCK];
  const i32 tid = threadIdx.x;

  // Calculate which output element this block computes
  // Total output elements = B × M
  const i32 total_outputs = B * M;
  const i32 start_idx = blockIdx.x * ROW_PER_BLOCK;
  const i32 end_idx = min(start_idx + ROW_PER_BLOCK, total_outputs);

  if (start_idx >= total_outputs) {
    return;
  }

  // Vectorization configuration
  constexpr i32 PACK_SIZE = 4;
  const i32 pack_num = N / PACK_SIZE;
  const i32 pack_off = pack_num * PACK_SIZE;

  // Process each output element assigned to this block
  #pragma unroll
  for (i32 idx = start_idx; idx < end_idx; ++idx) {
    // Decode output index to (batch, row)
    const i32 batch = idx / M;
    const i32 row = idx % M;

    // Initialize accumulator
    sdata[tid] = 0.0f;

    // Input row for this batch
    const i32 input_offset = batch * N;
    const float4* input_pack = reinterpret_cast<const float4*>(input + input_offset);

    // Weight row for this output
    const i32 weight_offset = row * N;
    const float4* weight_pack = reinterpret_cast<const float4*>(weight + weight_offset);

    // ============================================
    // Phase 1: Vectorized dot product
    // ============================================
    #pragma unroll
    for (i32 i = tid; i < pack_num; i += THREAD_PER_BLOCK) {
      float4 in_val = input_pack[i];
      float4 w_val = weight_pack[i];

      float partial_sum = in_val.x * w_val.x +
                         in_val.y * w_val.y +
                         in_val.z * w_val.z +
                         in_val.w * w_val.w;
      sdata[tid] += partial_sum;
    }

    // ============================================
    // Phase 2: Scalar tail
    // ============================================
    for (i32 i = pack_off + tid; i < N; i += THREAD_PER_BLOCK) {
      sdata[tid] += input[input_offset + i] * weight[weight_offset + i];
    }

    __syncthreads();

    // ============================================
    // Phase 3: Block-level reduction
    // ============================================
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sdata[tid]);
    __syncthreads();

    // Thread 0 writes final result
    if (tid == 0) {
      output[idx] = block_sum;
    }
    __syncthreads();
  }
}

// ============================================================================
// Host-side Launch Functions
// ============================================================================

Result<void> matmul_gemv_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 M,
    i32 N,
    cudaStream_t stream) {

  // Validate input sizes
  if (input.size() != static_cast<usize>(N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in matmul_gemv_cuda_launch");
  }
  if (weight.size() != static_cast<usize>(M * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in matmul_gemv_cuda_launch");
  }
  if (output.size() != static_cast<usize>(M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in matmul_gemv_cuda_launch");
  }

  // Check alignment for vectorization
  if (N % MATMUL_PACK_SIZE != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input dimension N must be multiple of 4 for vectorization");
  }

  // Launch configuration
  constexpr i32 BLOCK_SIZE = MATMUL_BLOCK_SIZE;
  constexpr i32 ROWS_PER_BLOCK = MATMUL_ROWS_PER_BLOCK;
  const i32 grid_size = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

  if (stream != nullptr) {
    matmul_gemv_kernel<BLOCK_SIZE, ROWS_PER_BLOCK><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input.data(), weight.data(), output.data(), M, N);
  } else {
    matmul_gemv_kernel<BLOCK_SIZE, ROWS_PER_BLOCK><<<grid_size, BLOCK_SIZE>>>(
        input.data(), weight.data(), output.data(), M, N);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA matmul_gemv kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

Result<void> matmul_gemm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 B,
    i32 M,
    i32 N,
    cudaStream_t stream) {

  // Validate input sizes
  if (input.size() != static_cast<usize>(B * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in matmul_gemm_cuda_launch");
  }
  if (weight.size() != static_cast<usize>(M * N)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in matmul_gemm_cuda_launch");
  }
  if (output.size() != static_cast<usize>(B * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in matmul_gemm_cuda_launch");
  }

  if (N % MATMUL_PACK_SIZE != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input dimension N must be multiple of 4 for vectorization");
  }

  // Launch configuration
  constexpr i32 BLOCK_SIZE = MATMUL_BLOCK_SIZE;
  constexpr i32 ROWS_PER_BLOCK = MATMUL_ROWS_PER_BLOCK;
  const i32 total_outputs = B * M;
  const i32 grid_size = (total_outputs + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

  if (stream != nullptr) {
    matmul_gemm_kernel<BLOCK_SIZE, ROWS_PER_BLOCK><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input.data(), weight.data(), output.data(), B, M, N);
  } else {
    matmul_gemm_kernel<BLOCK_SIZE, ROWS_PER_BLOCK><<<grid_size, BLOCK_SIZE>>>(
        input.data(), weight.data(), output.data(), B, M, N);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA matmul_gemm kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// Template Instantiation
// ============================================================================

template __global__ void matmul_gemv_kernel<MATMUL_BLOCK_SIZE, MATMUL_ROWS_PER_BLOCK>(
    const float*, const float*, float*, i32, i32);
template __global__ void matmul_gemm_kernel<MATMUL_BLOCK_SIZE, MATMUL_ROWS_PER_BLOCK>(
    const float*, const float*, float*, i32, i32, i32);

}  // namespace photon::kernels::cuda
