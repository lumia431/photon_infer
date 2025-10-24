/**
 * @file matmul_kernel.cuh
 * @brief CUDA matrix multiplication kernel interface
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/matmul_kernel.cu
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH

#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA matrix-vector multiplication (GEMV) kernel
 *
 * Following KuiperInfer's design:
 * - Computes: output[M] = input[N] @ weight[M×N]^T
 * - Each block computes 1 output element
 * - Uses float4 vectorization
 * - Uses CUB BlockReduce for efficient reduction
 * - Grid size = M, Block size = 128
 *
 * @param input Input vector [N]
 * @param weight Weight matrix [M × N]
 * @param output Output vector [M]
 * @param M Number of output elements (rows in weight)
 * @param N Input dimension (cols in weight)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> matmul_gemv_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 M,
    i32 N,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_MATMUL_KERNEL_CUH
