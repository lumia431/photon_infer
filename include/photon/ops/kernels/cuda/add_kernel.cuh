/**
 * @file add_kernel.cuh
 * @brief CUDA kernels for element-wise addition
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_KERNELS_CUDA_ADD_KERNEL_CUH
#define PHOTON_OPS_KERNELS_CUDA_ADD_KERNEL_CUH

#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

#include <span>
#include <cuda_runtime.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for element-wise addition with vectorization
 *
 * Computes: output[i] = input1[i] + input2[i]
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param input1 First input array
 * @param input2 Second input array
 * @param output Output array
 * @param size Number of elements
 */
template <i32 BLOCK_SIZE>
__global__ void add_kernel(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    i32 size);

/**
 * @brief Launch element-wise addition CUDA kernel
 *
 * @param input1 First input data (device)
 * @param input2 Second input data (device)
 * @param output Output data (device)
 * @param size Number of elements
 * @param stream CUDA stream
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
