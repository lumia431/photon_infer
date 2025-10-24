#include "photon/core/tensor.hpp"
#include "photon/ops/embedding.hpp"
#include <iostream>

using namespace photon;

int main() {
  std::cout << "Testing CUDA Embedding...\n";

  // Create embedding operator
  EmbeddingOp emb_op(100, 64);  // vocab=100, dim=64
  emb_op.set_device(DeviceType::CUDA);

  // Create weight on CPU
  auto weight_result = Tensor::zeros({100, 64}, DataType::Float32, DeviceType::CPU);
  if (!weight_result) {
    std::cerr << "Failed to create weight\n";
    return 1;
  }

  // Initialize some values
  auto weight_map = weight_result.value().matrix_map<f32>();
  weight_map(0, 0) = 1.0f;
  weight_map(1, 0) = 2.0f;

  // Set weight (should auto-convert to CUDA)
  auto set_result = emb_op.set_weight(std::move(weight_result.value()));
  if (!set_result) {
    std::cerr << "Failed to set weight: " << set_result.error().message() << "\n";
    return 1;
  }

  auto init_result = emb_op.init();
  if (!init_result) {
    std::cerr << "Failed to init: " << init_result.error().message() << "\n";
    return 1;
  }

  std::cout << "Embedding initialized\n";

  // Create input on CUDA
  auto input_result = Tensor::create({2}, DataType::Int32, DeviceType::CUDA);
  if (!input_result) {
    std::cerr << "Failed to create input\n";
    return 1;
  }

  // Copy indices to GPU
  i32 indices[2] = {0, 1};
  cudaError_t err = cudaMemcpy(input_result.value().data(), indices,
                               2 * sizeof(i32), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy indices: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  // Create output
  auto output_result = Tensor::create({2, 64}, DataType::Float32, DeviceType::CUDA);
  if (!output_result) {
    std::cerr << "Failed to create output\n";
    return 1;
  }

  std::cout << "Running forward...\n";
  auto fwd_result = emb_op.forward(input_result.value(), output_result.value());
  if (!fwd_result) {
    std::cerr << "Forward failed: " << fwd_result.error().message() << "\n";
    return 1;
  }

  std::cout << "Forward succeeded!\n";

  // Copy result back to check
  auto cpu_output = output_result.value().to(DeviceType::CPU);
  if (!cpu_output) {
    std::cerr << "Failed to copy to CPU: " << cpu_output.error().message() << "\n";
    return 1;
  }

  auto out_map = cpu_output.value().matrix_map<f32>();
  std::cout << "Output[0,0] = " << out_map(0, 0) << " (expected 1.0)\n";
  std::cout << "Output[1,0] = " << out_map(1, 0) << " (expected 2.0)\n";

  std::cout << "Test passed!\n";
  return 0;
}
