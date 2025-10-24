/**
 * @file swiglu_kernel.cuh
 * @brief CUDA SwiGLU activation kernel interface
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/swiglu_kernel.cu
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH

#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA SwiGLU activation kernel
 *
 * Following KuiperInfer's design:
 * - SwiGLU: out = swish(in1) * in2
 * - Swish: swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
 * - Uses shared memory for caching inputs
 * - Grid: (size + 128 - 1) / 128 blocks
 * - Block: 128 threads
 * - Shared memory: 128 * sizeof(float) * 2
 *
 * @param input1 First input tensor (gate values)
 * @param input2 Second input tensor (values to modulate)
 * @param output Output tensor
 * @param size Number of elements
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> swiglu_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_SWIGLU_KERNEL_CUH
