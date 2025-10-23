/**
 * @file add_kernel.cu
 * @brief CUDA implementation of element-wise addition
 * @version 0.1.0
 */

#include "photon/ops/kernels/cuda/add_kernel.cuh"

namespace photon::kernels::cuda {

template <i32 BLOCK_SIZE>
__global__ void add_kernel(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    i32 size) {

  const i32 tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Vectorized access with float4
  constexpr i32 PACK_SIZE = 4;
  const i32 pack_num = size / PACK_SIZE;
  const i32 pack_off = pack_num * PACK_SIZE;

  // Process vectorized part
  if (tid < pack_num) {
    const float4* in1_pack = reinterpret_cast<const float4*>(input1);
    const float4* in2_pack = reinterpret_cast<const float4*>(input2);
    float4* out_pack = reinterpret_cast<float4*>(output);

    float4 v1 = in1_pack[tid];
    float4 v2 = in2_pack[tid];

    out_pack[tid] = make_float4(
        v1.x + v2.x,
        v1.y + v2.y,
        v1.z + v2.z,
        v1.w + v2.w);
  }

  // Process scalar tail
  const i32 scalar_tid = pack_off + tid;
  if (scalar_tid < size) {
    output[scalar_tid] = input1[scalar_tid] + input2[scalar_tid];
  }
}

Result<void> add_cuda_launch(
    std::span<const f32> input1,
    std::span<const f32> input2,
    std::span<f32> output,
    i32 size,
    cudaStream_t stream) {

  if (input1.size() != static_cast<usize>(size) ||
      input2.size() != static_cast<usize>(size) ||
      output.size() != static_cast<usize>(size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Size mismatch in add_cuda_launch");
  }

  constexpr i32 BLOCK_SIZE = 256;
  const i32 grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (stream != nullptr) {
    add_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input1.data(), input2.data(), output.data(), size);
  } else {
    add_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
        input1.data(), input2.data(), output.data(), size);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA add kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

template __global__ void add_kernel<256>(
    const float*, const float*, float*, i32);

}  // namespace photon::kernels::cuda
