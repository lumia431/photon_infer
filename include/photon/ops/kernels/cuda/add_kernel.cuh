/**
 * @file add_kernel.cuh
 * @brief CUDA element-wise addition kernel interface
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/add_kernel.cu
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_ADD_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_ADD_KERNEL_CUH

#include <cuda_runtime.h>
#include <span>
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

namespace photon::kernels::cuda {

/**
 * @brief Launch CUDA element-wise addition kernel
 *
 * Following KuiperInfer's design:
 * - Each thread processes one element
 * - Grid: (size + 512 - 1) / 512 blocks
 * - Block: 512 threads
 *
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output Output tensor
 * @param size Number of elements
 * @param stream CUDA stream (optional)
 * @return Result indicating success or error
 */
Result<void> add_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream = nullptr);

}  // namespace photon::kernels::cuda

#endif  // PHOTON_OPS_KERNELS_CUDA_ADD_KERNEL_CUH
