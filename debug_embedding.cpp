/**
 * @file debug_embedding.cpp
 * @brief Debug CUDA Embedding step by step
 */

#include <iostream>
#include <cuda_runtime.h>
#include "photon/core/tensor.hpp"
#include "photon/ops/embedding.hpp"

using namespace photon;

int main() {
  std::cout << "=== Debugging CUDA Embedding ===\n\n";

  // Check CUDA availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA devices available!\n";
    return 1;
  }
  std::cout << "✓ CUDA devices found: " << device_count << "\n";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "  Using device: " << prop.name << "\n";
  std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n\n";

  // Test parameters
  const i32 vocab_size = 100;
  const i32 dim = 64;
  const i32 num_tokens = 3;

  std::cout << "Step 1: Creating CPU Embedding operator...\n";
  EmbeddingOp emb_cpu(vocab_size, dim);
  std::cout << "  ✓ CPU operator created\n";

  std::cout << "Step 2: Creating CUDA Embedding operator...\n";
  EmbeddingOp emb_cuda(vocab_size, dim);
  emb_cuda.set_device(DeviceType::CUDA);
  std::cout << "  ✓ CUDA operator created\n";

  std::cout << "Step 3: Creating weight tensor on CPU...\n";
  auto weight_result = Tensor::zeros({vocab_size, dim}, DataType::Float32, DeviceType::CPU);
  if (!weight_result) {
    std::cerr << "  ✗ Failed to create weight\n";
    return 1;
  }
  std::cout << "  ✓ Weight tensor created\n";

  // Initialize with simple pattern
  auto weight_map = weight_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size; ++i) {
    for (i32 j = 0; j < dim; ++j) {
      weight_map(i, j) = static_cast<f32>(i * 100 + j);
    }
  }
  std::cout << "  ✓ Weight initialized (pattern: i*100 + j)\n";
  std::cout << "  Example: weight[1][0] = " << weight_map(1, 0) << "\n";
  std::cout << "           weight[2][0] = " << weight_map(2, 0) << "\n\n";

  std::cout << "Step 4: Setting weight for CPU operator...\n";
  auto weight_cpu_result = Tensor::zeros({vocab_size, dim}, DataType::Float32, DeviceType::CPU);
  auto cpu_map = weight_cpu_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size; ++i) {
    for (i32 j = 0; j < dim; ++j) {
      cpu_map(i, j) = static_cast<f32>(i * 100 + j);
    }
  }
  if (!emb_cpu.set_weight(std::move(weight_cpu_result.value()))) {
    std::cerr << "  ✗ Failed to set CPU weight\n";
    return 1;
  }
  if (!emb_cpu.init()) {
    std::cerr << "  ✗ Failed to init CPU operator\n";
    return 1;
  }
  std::cout << "  ✓ CPU operator initialized\n";

  std::cout << "Step 5: Setting weight for CUDA operator...\n";
  auto weight_cuda_result = Tensor::zeros({vocab_size, dim}, DataType::Float32, DeviceType::CPU);
  auto cuda_map = weight_cuda_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size; ++i) {
    for (i32 j = 0; j < dim; ++j) {
      cuda_map(i, j) = static_cast<f32>(i * 100 + j);
    }
  }
  if (!emb_cuda.set_weight(std::move(weight_cuda_result.value()))) {
    std::cerr << "  ✗ Failed to set CUDA weight\n";
    return 1;
  }
  if (!emb_cuda.init()) {
    std::cerr << "  ✗ Failed to init CUDA operator\n";
    return 1;
  }
  std::cout << "  ✓ CUDA operator initialized\n\n";

  std::cout << "Step 6: Creating input tokens on CPU...\n";
  auto tokens_cpu = Tensor::create({num_tokens}, DataType::Int32, DeviceType::CPU);
  if (!tokens_cpu) {
    std::cerr << "  ✗ Failed to create CPU tokens\n";
    return 1;
  }
  i32* token_ptr = tokens_cpu.value().ptr<i32>();
  token_ptr[0] = 1;
  token_ptr[1] = 2;
  token_ptr[2] = 5;
  std::cout << "  ✓ Token IDs: [1, 2, 5]\n";

  std::cout << "Step 7: Copying tokens to CUDA...\n";
  auto tokens_cuda = Tensor::create({num_tokens}, DataType::Int32, DeviceType::CUDA);
  if (!tokens_cuda) {
    std::cerr << "  ✗ Failed to create CUDA tokens\n";
    return 1;
  }
  err = cudaMemcpy(tokens_cuda.value().data(), tokens_cpu.value().data(),
                   num_tokens * sizeof(i32), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "  ✗ Failed to copy tokens: " << cudaGetErrorString(err) << "\n";
    return 1;
  }
  std::cout << "  ✓ Tokens copied to GPU\n\n";

  std::cout << "Step 8: Creating output tensors...\n";
  auto out_cpu = Tensor::create({num_tokens, dim}, DataType::Float32, DeviceType::CPU);
  auto out_cuda = Tensor::create({num_tokens, dim}, DataType::Float32, DeviceType::CUDA);
  if (!out_cpu || !out_cuda) {
    std::cerr << "  ✗ Failed to create output tensors\n";
    return 1;
  }
  std::cout << "  ✓ Output tensors created\n\n";

  std::cout << "Step 9: Running CPU embedding...\n";
  auto cpu_result = emb_cpu.forward(tokens_cpu.value(), out_cpu.value());
  if (!cpu_result) {
    std::cerr << "  ✗ CPU forward failed: " << cpu_result.error().message() << "\n";
    return 1;
  }
  std::cout << "  ✓ CPU forward completed\n";

  auto cpu_out_map = out_cpu.value().matrix_map<f32>();
  std::cout << "  CPU output[0][0:3] = [" << cpu_out_map(0, 0) << ", "
            << cpu_out_map(0, 1) << ", " << cpu_out_map(0, 2) << "]\n";
  std::cout << "  Expected: [100, 101, 102] (token 1)\n\n";

  std::cout << "Step 10: Running CUDA embedding...\n";
  auto cuda_result = emb_cuda.forward(tokens_cuda.value(), out_cuda.value());
  if (!cuda_result) {
    std::cerr << "  ✗ CUDA forward failed: " << cuda_result.error().message() << "\n";
    return 1;
  }
  std::cout << "  ✓ CUDA forward completed\n";

  // Copy back to CPU for comparison
  auto out_cuda_cpu = out_cuda.value().to(DeviceType::CPU);
  if (!out_cuda_cpu) {
    std::cerr << "  ✗ Failed to copy CUDA output to CPU\n";
    return 1;
  }
  auto cuda_out_map = out_cuda_cpu.value().matrix_map<f32>();
  std::cout << "  CUDA output[0][0:3] = [" << cuda_out_map(0, 0) << ", "
            << cuda_out_map(0, 1) << ", " << cuda_out_map(0, 2) << "]\n";
  std::cout << "  Expected: [100, 101, 102] (token 1)\n\n";

  std::cout << "Step 11: Comparing CPU vs CUDA outputs...\n";
  bool all_match = true;
  for (i32 i = 0; i < num_tokens; ++i) {
    for (i32 j = 0; j < dim; ++j) {
      f32 cpu_val = cpu_out_map(i, j);
      f32 cuda_val = cuda_out_map(i, j);
      if (std::abs(cpu_val - cuda_val) > 1e-5f) {
        std::cerr << "  ✗ Mismatch at [" << i << "][" << j << "]: "
                  << "CPU=" << cpu_val << " CUDA=" << cuda_val << "\n";
        all_match = false;
        break;
      }
    }
    if (!all_match) break;
  }

  if (all_match) {
    std::cout << "  ✓ All outputs match!\n\n";
    std::cout << "=== ✓ CUDA Embedding Test PASSED ===\n";
    return 0;
  } else {
    std::cout << "\n=== ✗ CUDA Embedding Test FAILED ===\n";
    return 1;
  }
}
