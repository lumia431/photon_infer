/**
 * @file matmul_kernel.cu
 * @brief CUDA matrix multiplication kernel implementation
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/matmul_kernel.cu
 */

#include "photon/ops/kernels/cuda/matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for matrix-vector multiplication (GEMV)
 *
 * Following KuiperInfer line-by-line:
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

  // Each block processes ROW_PER_BLOCK rows (following KuiperInfer)
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // Vectorization configuration (following KuiperInfer)
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  // Process each row (following KuiperInfer)
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

    // Vectorized dot product (following KuiperInfer)
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

    // Handle remaining elements (following KuiperInfer)
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    // Block-level reduction using CUB (following KuiperInfer)
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    // Thread 0 writes result (following KuiperInfer)
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

  // Validate dimensions (following KuiperInfer)
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

  // Check vectorization alignment (following KuiperInfer)
  constexpr int packet_size = 4;
  if (M % packet_size != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input dimension M must be multiple of 4 for vectorization");
  }

  // Launch configuration (following KuiperInfer exactly)
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

}  // namespace photon::kernels::cuda
