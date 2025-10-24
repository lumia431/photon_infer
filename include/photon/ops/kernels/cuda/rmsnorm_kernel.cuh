/**
 * @file rmsnorm_kernel.cuh
 * @brief CUDA RMS normalization kernel interface
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/rmsnorm_kernel.cu
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH

#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA RMS normalization kernel
 *
 * Following KuiperInfer's design:
 * - Computes: output = input * weight / sqrt(mean(input^2) + eps)
 * - Uses CUB BlockReduce for sum of squares
 * - Uses float4 vectorization
 * - Grid: 1 block, Block: 128 threads
 * - eps = 1e-5f
 *
 * @param input Input tensor [dim]
 * @param weight Weight tensor [dim]
 * @param output Output tensor [dim]
 * @param dim Dimension size
 * @param eps Epsilon for numerical stability (default 1e-5f)
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps = 1e-5f,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_RMSNORM_KERNEL_CUH
