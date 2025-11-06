/**
 * @file matmul_cublaslt.cu
 * @brief cuBLASLt-based INT8 matrix multiplication implementation
 * @version 1.0.0
 *
 * Strategy:
 * Since cuBLASLt doesn't natively support per-group quantization dequantization,
 * we use a hybrid approach:
 * 1. Fast dequantization kernel: INT8 + scales -> FP32
 * 2. cuBLAS SGEMM: FP32 @ FP32^T -> FP32 (highly optimized)
 *
 * This is still much faster than custom INT8 GEMV/GEMM kernels because:
 * - Dequantization is memory-bound and can be heavily optimized
 * - cuBLAS SGEMM is extremely optimized (close to peak FLOPS)
 * - Overall time: dequant + GEMM < custom INT8 GEMM
 */

#include "photon/ops/kernels/cuda/matmul_cublaslt.cuh"
#include <cublas_v2.h>
#include <glog/logging.h>

namespace photon::kernels::cuda {

// ============================================================================
// cuBLAS Handle Management
// ============================================================================

// Global cuBLAS handle (initialized once)
static cublasHandle_t g_cublas_handle = nullptr;

Result<cublasLtHandle_t> cublaslt_init() {
  if (g_cublas_handle != nullptr) {
    // Already initialized, return existing handle
    return Ok(reinterpret_cast<cublasLtHandle_t>(g_cublas_handle));
  }

  cublasStatus_t status = cublasCreate(&g_cublas_handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return Err<cublasLtHandle_t>(ErrorCode::CudaError,
                                 "Failed to create cuBLAS handle");
  }

  return Ok(reinterpret_cast<cublasLtHandle_t>(g_cublas_handle));
}

void cublaslt_cleanup(cublasLtHandle_t handle) {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
}

// ============================================================================
// Fast Dequantization Kernel
// ============================================================================

/**
 * @brief Vectorized dequantization kernel: INT8 + scales -> FP32
 *
 * Uses float4 vectorized loads/stores for maximum memory bandwidth.
 * Each thread dequantizes 4 elements at once.
 *
 * Grid: (total_elements / 4 / 256) blocks
 * Block: 256 threads
 */
__global__ void dequantize_int8_kernel(
    const i8* __restrict__ weight_int8,
    const f32* __restrict__ scales,
    f32* __restrict__ weight_fp32,
    i32 total_elements,
    i32 group_size) {

  const i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  const i32 stride = blockDim.x * gridDim.x;

  // Process 4 elements per thread using vectorized loads
  for (i32 i = tid * 4; i < total_elements; i += stride * 4) {
    // Bounds check
    if (i + 3 >= total_elements) {
      // Handle remainder elements one by one
      for (i32 j = i; j < total_elements; ++j) {
        i32 group_idx = j / group_size;
        f32 scale = scales[group_idx];
        weight_fp32[j] = static_cast<f32>(weight_int8[j]) * scale;
      }
      break;
    }

    // Vectorized processing (4 elements at once)
    // Load 4 INT8 values
    int8_t val0 = weight_int8[i + 0];
    int8_t val1 = weight_int8[i + 1];
    int8_t val2 = weight_int8[i + 2];
    int8_t val3 = weight_int8[i + 3];

    // Calculate group indices
    i32 group_idx0 = (i + 0) / group_size;
    i32 group_idx1 = (i + 1) / group_size;
    i32 group_idx2 = (i + 2) / group_size;
    i32 group_idx3 = (i + 3) / group_size;

    // Load scales (will be cached if same group)
    f32 scale0 = scales[group_idx0];
    f32 scale1 = scales[group_idx1];
    f32 scale2 = scales[group_idx2];
    f32 scale3 = scales[group_idx3];

    // Dequantize and store
    weight_fp32[i + 0] = static_cast<f32>(val0) * scale0;
    weight_fp32[i + 1] = static_cast<f32>(val1) * scale1;
    weight_fp32[i + 2] = static_cast<f32>(val2) * scale2;
    weight_fp32[i + 3] = static_cast<f32>(val3) * scale3;
  }
}

/**
 * @brief Launch dequantization kernel
 */
Result<void> dequantize_int8_launch(
    const i8* weight_int8,
    const f32* scales,
    f32* weight_fp32,
    i32 total_elements,
    i32 group_size,
    cudaStream_t stream) {

  // Launch configuration: 256 threads per block, process 4 elements per thread
  const int threads = 256;
  const int elements_per_thread = 4;
  const int blocks = (total_elements + threads * elements_per_thread - 1) /
                     (threads * elements_per_thread);

  dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(
      weight_int8, scales, weight_fp32, total_elements, group_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("Dequantization kernel launch failed: ") +
                    cudaGetErrorString(err));
  }

  return Ok();
}

// ============================================================================
// INT8 GEMV with cuBLAS (via dequantization)
// ============================================================================

Result<void> matmul_gemv_cublaslt_launch(
    cublasLtHandle_t handle_lt,
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

  // Cast handle back to cublasHandle_t
  cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(handle_lt);

  // Set stream
  if (stream != nullptr) {
    cublasSetStream(handle, stream);
  }

  // Allocate temporary buffer for dequantized weight
  f32* weight_fp32 = nullptr;
  cudaError_t err = cudaMalloc(&weight_fp32, K * M * sizeof(f32));
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::OutOfMemory,
                    "Failed to allocate dequantization buffer");
  }

  // Step 1: Dequantize weight (INT8 -> FP32)
  auto dequant_result = dequantize_int8_launch(
      weight_ptr, scales_ptr, weight_fp32, K * M, group_size, stream);

  if (!dequant_result) {
    cudaFree(weight_fp32);
    return dequant_result;
  }

  // Step 2: cuBLAS GEMV: output[K] = weight[K×M] @ input[M]
  // Matrix layout: weight is [K×M] row-major
  // cuBLAS expects column-major, so we compute: output = weight * input
  //
  // cublasSgemv parameters:
  // - handle: cuBLAS handle
  // - trans: CUBLAS_OP_N (no transpose, use weight as-is but interpret as col-major)
  // - m: number of rows of matrix (K)
  // - n: number of columns of matrix (M)
  // - alpha: scalar multiplier (1.0)
  // - A: matrix pointer (weight_fp32)
  // - lda: leading dimension (K for col-major interpretation)
  // - x: input vector (input_ptr)
  // - incx: stride (1)
  // - beta: scalar multiplier for output (0.0, overwrite)
  // - y: output vector (output_ptr)
  // - incy: stride (1)

  const f32 alpha = 1.0f;
  const f32 beta = 0.0f;

  // Note: We store weight in row-major [K×M], but cuBLAS expects col-major.
  // To avoid transpose, we compute: y = A*x where A is [K×M] interpreted as col-major
  // This effectively computes: output[K] = weight[K×M] @ input[M]
  cublasStatus_t status = cublasSgemv(
      handle,
      CUBLAS_OP_T,  // Transpose because we're row-major but cuBLAS expects col-major
      M,            // Number of rows in weight (before transpose)
      K,            // Number of cols in weight (before transpose)
      &alpha,
      weight_fp32,  // Weight matrix [K×M] in row-major
      M,            // Leading dimension (row-major stride)
      input_ptr,    // Input vector [M]
      1,            // Input stride
      &beta,
      output_ptr,   // Output vector [K]
      1);           // Output stride

  // Free temporary buffer
  cudaFree(weight_fp32);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return Err<void>(ErrorCode::CudaError,
                    "cuBLAS GEMV failed with status: " +
                    std::to_string(static_cast<int>(status)));
  }

  return Ok();
}

// ============================================================================
// Batched INT8 GEMM with cuBLAS
// ============================================================================

Result<void> matmul_gemm_cublaslt_launch(
    cublasLtHandle_t handle_lt,
    const f32* input_ptr,
    usize input_size,
    const i8* weight_ptr,
    usize weight_size,
    const f32* scales_ptr,
    usize scales_size,
    i32 group_size,
    f32* output_ptr,
    usize output_size,
    i32 B,
    i32 K,
    i32 M,
    cudaStream_t stream) {

  // Cast handle back to cublasHandle_t
  cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(handle_lt);

  // Set stream
  if (stream != nullptr) {
    cublasSetStream(handle, stream);
  }

  // Allocate temporary buffer for dequantized weight (shared across batch)
  f32* weight_fp32 = nullptr;
  cudaError_t err = cudaMalloc(&weight_fp32, K * M * sizeof(f32));
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::OutOfMemory,
                    "Failed to allocate dequantization buffer");
  }

  // Step 1: Dequantize weight once (reused for all batch elements)
  auto dequant_result = dequantize_int8_launch(
      weight_ptr, scales_ptr, weight_fp32, K * M, group_size, stream);

  if (!dequant_result) {
    cudaFree(weight_fp32);
    return dequant_result;
  }

  // Step 2: cuBLAS batched GEMM: output[B×K] = input[B×M] @ weight[K×M]^T
  //
  // Matrix shapes:
  // - input: [B×M] (row-major)
  // - weight: [K×M] (row-major)
  // - output: [B×K] (row-major)
  //
  // cuBLAS GEMM: C = alpha * op(A) * op(B) + beta * C
  // We want: output[B×K] = input[B×M] @ weight^T[M×K]
  //
  // Mapping (interpreting row-major as col-major):
  // - A = weight^T [K×M] -> becomes [M×K] after transpose
  // - B = input [B×M]
  // - C = output [B×K]
  //
  // cublasSgemm parameters:
  // - transa: CUBLAS_OP_T (transpose weight)
  // - transb: CUBLAS_OP_N (no transpose input)
  // - m: number of rows of C (B)
  // - n: number of cols of C (K)
  // - k: inner dimension (M)

  const f32 alpha = 1.0f;
  const f32 beta = 0.0f;

  cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_N,  // No transpose for weight (but cuBLAS is col-major)
      CUBLAS_OP_N,  // No transpose for input
      K,            // Number of rows of weight^T = cols of output
      B,            // Number of rows of input = rows of output
      M,            // Inner dimension
      &alpha,
      weight_fp32,  // Weight [K×M] row-major
      K,            // Leading dimension of weight
      input_ptr,    // Input [B×M] row-major
      M,            // Leading dimension of input
      &beta,
      output_ptr,   // Output [B×K] row-major
      K);           // Leading dimension of output

  // Free temporary buffer
  cudaFree(weight_fp32);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return Err<void>(ErrorCode::CudaError,
                    "cuBLAS GEMM failed with status: " +
                    std::to_string(static_cast<int>(status)));
  }

  return Ok();
}

}  // namespace photon::kernels::cuda
