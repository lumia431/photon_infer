/**
 * @file matmul_kernel_quant.cu
 * @brief CUDA kernels for quantized (int8) matrix multiplication
 * @version 0.1.0
 *
 * This file implements int8 quantized matrix-vector multiplication with
 * dynamic dequantization using group-wise symmetric quantization.
 *
 * Key features:
 * - int8 weights with float32 scales (per-group)
 * - Dynamic dequantization during computation
 * - Optimized using CUB block-level primitives
 * - Inspired by KuiperInfer's implementation
 */

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

// ============================================================================
// CUDA Kernel: Quantized Matrix-Vector Multiplication
// ============================================================================

/**
 * @brief CUDA kernel for quantized GEMV with dynamic dequantization
 *
 * Computes: output[K] = input[M] @ weight_int8[K × M]^T
 * where weight is stored as int8 with per-group scales.
 *
 * Algorithm:
 * 1. Each block processes one output element (row of weight matrix)
 * 2. Each thread computes partial sum for multiple input elements
 * 3. Dequantize on-the-fly: dequant = scale[group_idx] * int8_weight
 * 4. Use CUB BlockReduce for efficient parallel reduction
 *
 * Thread Organization:
 * - Grid: [K blocks] (one block per output element)
 * - Block: [THREADS_PER_BLOCK threads]
 * - Each thread processes multiple elements (stride loop)
 *
 * Memory Access Pattern:
 * - Input: coalesced read (all threads access contiguous elements)
 * - Weight: coalesced read (row-major layout)
 * - Scales: cached in shared memory / registers
 *
 * Template Parameters:
 * @tparam THREADS_PER_BLOCK Number of threads per block (typically 128-256)
 * @tparam ROWS_PER_BLOCK Number of output rows per block (usually 1)
 *
 * Kernel Parameters:
 * @param input Input activations [M] (float32)
 * @param weight Quantized weights [K × M] (int8, row-major)
 * @param scales Per-group scale factors (float32)
 * @param group_size Number of elements per quantization group
 * @param output Output activations [K] (float32)
 * @param M Input dimension (number of columns)
 * @param K Output dimension (number of rows)
 */
template <int THREADS_PER_BLOCK, int ROWS_PER_BLOCK>
__global__ void matmul_gemv_quant_kernel(
    const f32* __restrict__ input,
    const i8* __restrict__ weight,
    const f32* __restrict__ scales,
    const i32 group_size,
    f32* __restrict__ output,
    const i32 M,
    const i32 K) {

  // Shared memory for block-level reduction
  __shared__ f32 sdata[THREADS_PER_BLOCK];

  const u32 tid = threadIdx.x;
  const i32 start_row = blockIdx.x * ROWS_PER_BLOCK;
  const i32 end_row = start_row + ROWS_PER_BLOCK;

  // Early exit for out-of-bounds blocks
  if (start_row >= K) {
    return;
  }

  // Process each output row assigned to this block
  for (i32 row = start_row; row < end_row && row < K; ++row) {
    // Initialize thread-local accumulator
    f32 thread_sum = 0.0f;

    const i32 row_offset = row * M;

    // Stride loop: each thread processes multiple elements
    // This improves ILP (Instruction-Level Parallelism)
    for (i32 col = tid; col < M; col += THREADS_PER_BLOCK) {
      const i32 weight_idx = row_offset + col;
      const i32 group_idx = weight_idx / group_size;

      // Load data (coalesced memory access)
      const f32 input_val = input[col];
      const i8 weight_val = weight[weight_idx];
      const f32 scale = scales[group_idx];

      // Compute: input * scale * weight (dynamic dequantization)
      // Compiler will fuse these operations
      thread_sum += input_val * scale * static_cast<f32>(weight_val);
    }

    // Store partial sum in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction using CUB (optimized)
    using BlockReduce = cub::BlockReduce<f32, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    f32 block_sum = BlockReduce(temp_storage).Sum(sdata[tid]);
    __syncthreads();

    // Write result (only thread 0 writes)
    if (tid == 0) {
      output[row] = block_sum;
    }
    __syncthreads();
  }
}

// ============================================================================
// Kernel Launch Wrapper
// ============================================================================

/**
 * @brief Launch quantized GEMV kernel with error checking
 *
 * This wrapper:
 * - Validates input dimensions
 * - Configures optimal grid/block dimensions
 * - Launches kernel with error checking
 *
 * @param input_ptr Input tensor data [M]
 * @param input_size Size of input
 * @param weight_ptr Quantized weight tensor data [K × M]
 * @param weight_size Size of weight
 * @param scales_ptr Scale factors [num_groups]
 * @param scales_size Size of scales
 * @param group_size Group size for quantization
 * @param output_ptr Output tensor data [K]
 * @param output_size Size of output
 * @param M Input dimension
 * @param K Output dimension
 * @param stream CUDA stream (nullptr for default stream)
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_quant_launch(
    const f32* input_ptr,
    usize input_size,
    const i8* weight_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 M,
    i32 K,
    cudaStream_t stream) {

  // Validate dimensions
  if (input_size != static_cast<usize>(M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in quantized matmul");
  }

  if (weight_size != static_cast<usize>(K * M)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in quantized matmul");
  }

  if (output_size != static_cast<usize>(K)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in quantized matmul");
  }

  // Kernel configuration
  // Following KuiperInfer: 128 threads per block, 1 row per block
  constexpr int THREADS = 128;
  constexpr int ROWS_PER_BLOCK = 1;

  const int num_blocks = K;

  // Launch kernel
  if (stream != nullptr) {
    matmul_gemv_quant_kernel<THREADS, ROWS_PER_BLOCK>
        <<<num_blocks, THREADS, 0, stream>>>(
            input_ptr, weight_ptr, scales_ptr, group_size, output_ptr, M, K);
  } else {
    matmul_gemv_quant_kernel<THREADS, ROWS_PER_BLOCK>
        <<<num_blocks, THREADS>>>(
            input_ptr, weight_ptr, scales_ptr, group_size, output_ptr, M, K);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    "CUDA kernel launch failed: " +
                    std::string(cudaGetErrorString(err)));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
