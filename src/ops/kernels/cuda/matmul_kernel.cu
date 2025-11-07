/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file matmul_kernel.cu
 * @brief CUDA matrix multiplication kernel implementation
 * @version 0.1.0
 *
 * Implementation based on standard practices at:
 * 
 */

#include "photon/ops/kernels/cuda/matmul_kernel.cuh"
#include "photon/ops/kernels/cuda/matmul_gemv_quant.cuh"
#include <cub/block/block_reduce.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for matrix-vector multiplication (GEMV)
 *
 * Standard implementation:
 * - Template parameters: THREAD_PER_BLOCK=128, ROW_PER_BLOCK=1
 * - Each block computes ROW_PER_BLOCK output elements
 * - Uses float4 vectorization for efficiency
 * - Uses CUB BlockReduce for reduction
 *
 * Grid: K blocks, Block: 128 threads
 * Computes: output[K] = input[M] @ weight[K×M]^T
 */
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K) {

  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // Each block processes ROW_PER_BLOCK rows (using standard approach)
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // Vectorization configuration (using standard approach)
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  // Process each row (using standard approach)
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

    // Vectorized dot product (using standard approach)
#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x +
                       input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z +
                       input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    // Handle remaining elements (using standard approach)
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // Block-level reduction using CUB (using standard approach)
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    // Thread 0 writes result (using standard approach)
    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

Result<void> matmul_gemv_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 M,
    i32 N,
    cudaStream_t stream) {

  // Validate dimensions (using standard approach)
  if (static_cast<i32>(input.size()) != M) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in matmul_gemv_cuda_launch");
  }

  if (static_cast<i32>(weight.size()) != N * M) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in matmul_gemv_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != N) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in matmul_gemv_cuda_launch");
  }

  // Check vectorization alignment (using standard approach)
  constexpr int packet_size = 4;
  if (M % packet_size != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input dimension M must be multiple of 4 for vectorization");
  }

  // Launch configuration (using standard approach exactly)
  // Template: <THREAD_PER_BLOCK=128, ROW_PER_BLOCK=1>
  // Grid: N blocks, Block: 128 threads
  const i32 K = N;  // Number of output elements
  if (stream) {
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, stream>>>(
        input.data(), weight.data(), output.data(), M, K);
  } else {
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(
        input.data(), weight.data(), output.data(), M, K);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA matmul kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA matmul kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// cuBLAS Batched GEMM
// ============================================================================

#include <cublas_v2.h>

// Global cuBLAS handle (initialized on first use)
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_cublas_initialized = false;

static Result<cublasHandle_t> get_cublas_handle() {
  if (!g_cublas_initialized) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return Err<cublasHandle_t>(ErrorCode::CudaError,
                                 "Failed to create cuBLAS handle");
    }
    g_cublas_initialized = true;
  }
  return Ok(g_cublas_handle);
}

Result<void> matmul_gemm_cublas_launch(
    const f32* input,
    const f32* weight,
    f32* output,
    i32 batch_size,
    i32 M,
    i32 N,
    cudaStream_t stream) {

  // Get cuBLAS handle
  auto handle_result = get_cublas_handle();
  if (!handle_result) return Err<void>(handle_result.error());
  cublasHandle_t handle = handle_result.value();

  // Set stream if provided
  if (stream) {
    cublasSetStream(handle, stream);
  }

  const f32 alpha = 1.0f;
  const f32 beta = 0.0f;

  // We want (in row-major): output[batch, M] = input[batch, N] @ weight[M, N]^T
  //
  // Our tensors are row-major:
  // - input: [batch × N] stored as row-major
  // - weight: [M × N] stored as row-major
  // - output: [batch × M] stored as row-major
  //
  // cuBLAS expects column-major, so we transpose everything:
  // output^T[M × batch] = (weight[M × N]^T)^T @ input^T[N × batch]
  //                     = weight[N × M] @ input^T[N × batch]  (with transpose flags)
  //
  // In cuBLAS notation (column-major):
  // C[M × batch] = op(A)[M × N] @ op(B)[N × batch]
  // where op(A) = A^T, op(B) = B^T
  //
  // Since our row-major A is cuBLAS's column-major A^T:
  // C = weight^T @ input^T (with appropriate transpose flags)

  cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_T,  // Transpose weight: [M×N]^T = [N×M] in col-major
      CUBLAS_OP_T,  // Transpose input: [batch×N]^T = [N×batch] in col-major
      M,            // m: rows of output (col-major) = cols of output (row-major)
      batch_size,   // n: cols of output (col-major) = rows of output (row-major)
      N,            // k: inner dimension
      &alpha,
      weight,       // A: row-major [M×N] = col-major [N×M]
      N,            // lda: leading dim of A (row-major stride)
      input,        // B: row-major [batch×N] = col-major [N×batch]
      N,            // ldb: leading dim of B (row-major stride)
      &beta,
      output,       // C: row-major [batch×M] = col-major [M×batch]
      M);           // ldc: leading dim of C (row-major stride)

  if (status != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "cuBLAS GEMM failed with status: " << status;
    return Err<void>(ErrorCode::CudaError,
                    "cuBLAS GEMM failed");
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
