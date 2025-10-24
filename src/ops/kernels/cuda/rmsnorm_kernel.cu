/**
 * @file rmsnorm_kernel.cu
 * @brief CUDA RMS normalization kernel implementation
 * @version 0.1.0
 *
 * Strictly follows KuiperInfer implementation at:
 * demos/kuiper_llama/kuiper/source/op/kernels/cuda/rmsnorm_kernel.cu
 */

#include "photon/ops/kernels/cuda/rmsnorm_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <glog/logging.h>

namespace photon::kernels::cuda {

/**
 * @brief CUDA kernel for RMS normalization
 *
 * Following KuiperInfer line-by-line:
 * - Template parameter: BLOCK_DIM=128
 * - Single block, processes entire row
 * - Phase 1: Compute sum of squares using CUB BlockReduce
 * - Phase 2: Compute scale = rsqrt(mean + eps)
 * - Phase 3: Apply normalization with weight
 *
 * Grid: 1 block, Block: 128 threads
 */
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(
    float* in,
    float* wei,
    float* out,
    int size,
    float eps) {

  const int tid = threadIdx.x;

  // Vectorization configuration (following KuiperInfer)
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  // Phase 1: Compute sum of squares (following KuiperInfer)
  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  // Handle remaining elements (following KuiperInfer)
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  // Block-level reduction using CUB (following KuiperInfer)
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  // Compute scale factor (following KuiperInfer)
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // Phase 2: Apply normalization with weight (following KuiperInfer)
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x,
        scale * in_float4.y * wei_float4.y,
        scale * in_float4.z * wei_float4.z,
        scale * in_float4.w * wei_float4.w);
  }

  // Handle remaining elements (following KuiperInfer)
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

Result<void> rmsnorm_cuda_launch(
    std::span<const f32> input,
    std::span<const f32> weight,
    std::span<f32> output,
    i32 dim,
    f32 eps,
    cudaStream_t stream) {

  // Validate dimensions (following KuiperInfer)
  if (static_cast<i32>(input.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Input size mismatch in rmsnorm_cuda_launch");
  }

  if (static_cast<i32>(weight.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Weight size mismatch in rmsnorm_cuda_launch");
  }

  if (static_cast<i32>(output.size()) != dim) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Output size mismatch in rmsnorm_cuda_launch");
  }

  // Check vectorization alignment (following KuiperInfer)
  constexpr int pack_size = 4;
  if (dim % pack_size != 0) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Dimension must be multiple of 4 for vectorization");
  }

  // Launch configuration (following KuiperInfer exactly)
  // Grid: 1 block, Block: 128 threads
  constexpr int threads_num = 128;

  // Need non-const pointers for kernel (following KuiperInfer)
  float* in_ptr = const_cast<float*>(input.data());
  float* wei_ptr = const_cast<float*>(weight.data());
  float* out_ptr = output.data();

  if (stream) {
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream>>>(
        in_ptr, wei_ptr, out_ptr, dim, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(
        in_ptr, wei_ptr, out_ptr, dim, eps);
  }

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(ERROR) << "CUDA rmsnorm kernel launch failed: " << cudaGetErrorString(err);
    return Err<void>(ErrorCode::CudaError,
                    std::string("CUDA rmsnorm kernel launch failed: ") +
                        cudaGetErrorString(err));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
