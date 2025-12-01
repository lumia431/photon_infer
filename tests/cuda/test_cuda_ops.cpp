/**
 * @file test_cuda_ops.cpp
 * @brief CUDA operator validation tests
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "photon/core/tensor.hpp"
#include "photon/ops/embedding.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/ops/rmsnorm.hpp"

using namespace photon;

// Helper function to check if CUDA is available
bool is_cuda_available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

// Test CUDA Embedding
TEST(CUDAOpsTest, EmbeddingForward) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  constexpr i32 vocab_size = 128;
  constexpr i32 embedding_dim = 64;
  constexpr i32 num_tokens = 4;

  // Create CPU tensors
  auto weight_cpu_result = Tensor::create({vocab_size, embedding_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cpu_result);
  auto weight_cpu = std::move(weight_cpu_result.value());

  // Initialize weights with simple pattern
  for (i32 i = 0; i < vocab_size; ++i) {
    for (i32 j = 0; j < embedding_dim; ++j) {
      weight_cpu.ptr<f32>()[i * embedding_dim + j] = static_cast<f32>(i * 100 + j);
    }
  }

  // Create CUDA weight tensor
  auto weight_cuda_result = Tensor::create({vocab_size, embedding_dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(weight_cuda_result);
  auto weight_cuda = std::move(weight_cuda_result.value());

  // Copy weights to CUDA
  cudaError_t err = cudaMemcpy(weight_cuda.data(), weight_cpu.data(),
                               weight_cpu.size() * sizeof(f32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

  // Create input tokens on CPU
  auto tokens_cpu_result = Tensor::create({num_tokens}, DataType::Int32, DeviceType::CPU);
  ASSERT_TRUE(tokens_cpu_result);
  auto tokens_cpu = std::move(tokens_cpu_result.value());
  tokens_cpu.ptr<i32>()[0] = 0;
  tokens_cpu.ptr<i32>()[1] = 1;
  tokens_cpu.ptr<i32>()[2] = 5;
  tokens_cpu.ptr<i32>()[3] = 10;

  // Copy tokens to CUDA
  auto tokens_cuda_result = Tensor::create({num_tokens}, DataType::Int32, DeviceType::CUDA);
  ASSERT_TRUE(tokens_cuda_result);
  auto tokens_cuda = std::move(tokens_cuda_result.value());
  err = cudaMemcpy(tokens_cuda.data(), tokens_cpu.data(),
                   num_tokens * sizeof(i32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // Create output tensors
  auto output_cpu_result = Tensor::create({num_tokens, embedding_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cpu_result);
  auto output_cpu = std::move(output_cpu_result.value());

  auto output_cuda_result = Tensor::create({num_tokens, embedding_dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(output_cuda_result);
  auto output_cuda = std::move(output_cuda_result.value());

  // Create and initialize CPU operator
  EmbeddingOp op_cpu(vocab_size, embedding_dim);
  op_cpu.set_device(DeviceType::CPU);
  auto init_result = op_cpu.set_weight(std::move(weight_cpu));
  ASSERT_TRUE(init_result);
  init_result = op_cpu.init();
  ASSERT_TRUE(init_result);

  // Create and initialize CUDA operator
  EmbeddingOp op_cuda(vocab_size, embedding_dim);
  op_cuda.set_device(DeviceType::CUDA);
  init_result = op_cuda.set_weight(std::move(weight_cuda));
  ASSERT_TRUE(init_result);
  init_result = op_cuda.init();
  ASSERT_TRUE(init_result);

  // Run CPU forward
  auto forward_result = op_cpu.forward(tokens_cpu, output_cpu);
  ASSERT_TRUE(forward_result) << "CPU forward failed: " << forward_result.error().message();

  // Run CUDA forward
  forward_result = op_cuda.forward(tokens_cuda, output_cuda);
  ASSERT_TRUE(forward_result) << "CUDA forward failed: " << forward_result.error().message();

  // Synchronize CUDA
  err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess) << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err);

  // Copy CUDA output back to CPU for comparison
  auto output_cuda_cpu_result = Tensor::create({num_tokens, embedding_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cuda_cpu_result);
  auto output_cuda_cpu = std::move(output_cuda_cpu_result.value());
  err = cudaMemcpy(output_cuda_cpu.data(), output_cuda.data(),
                   output_cuda.size() * sizeof(f32), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Compare results
  const f32* cpu_data = output_cpu.ptr<f32>();
  const f32* cuda_data = output_cuda_cpu.ptr<f32>();
  for (usize i = 0; i < output_cpu.size(); ++i) {
    EXPECT_FLOAT_EQ(cpu_data[i], cuda_data[i])
        << "Mismatch at index " << i << ": CPU=" << cpu_data[i]
        << " CUDA=" << cuda_data[i];
  }

  std::cout << "✅ CUDA Embedding test passed!" << std::endl;
}

// Test CUDA MatMul
TEST(CUDAOpsTest, MatMulGEMV) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  constexpr i32 input_dim = 64;   // Must be multiple of 4
  constexpr i32 output_dim = 32;

  // Create CPU weight tensor
  auto weight_cpu_result = Tensor::create({output_dim, input_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cpu_result);
  auto weight_cpu = std::move(weight_cpu_result.value());

  // Initialize weights with simple pattern
  for (i32 i = 0; i < output_dim * input_dim; ++i) {
    weight_cpu.ptr<f32>()[i] = static_cast<f32>(i % 100) * 0.01f;
  }

  // Create CUDA weight tensor and copy
  auto weight_cuda_result = Tensor::create({output_dim, input_dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(weight_cuda_result);
  auto weight_cuda = std::move(weight_cuda_result.value());
  cudaError_t err = cudaMemcpy(weight_cuda.data(), weight_cpu.data(),
                               weight_cpu.size() * sizeof(f32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // Create input tensors
  auto input_cpu_result = Tensor::create({input_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input_cpu_result);
  auto input_cpu = std::move(input_cpu_result.value());
  for (i32 i = 0; i < input_dim; ++i) {
    input_cpu.ptr<f32>()[i] = static_cast<f32>(i) * 0.1f;
  }

  auto input_cuda_result = Tensor::create({input_dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(input_cuda_result);
  auto input_cuda = std::move(input_cuda_result.value());
  err = cudaMemcpy(input_cuda.data(), input_cpu.data(),
                   input_dim * sizeof(f32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // Create output tensors
  auto output_cpu_result = Tensor::create({output_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cpu_result);
  auto output_cpu = std::move(output_cpu_result.value());

  auto output_cuda_result = Tensor::create({output_dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(output_cuda_result);
  auto output_cuda = std::move(output_cuda_result.value());

  // Create and initialize operators
  MatMulOp op_cpu(input_dim, output_dim);
  op_cpu.set_device(DeviceType::CPU);
  auto init_result = op_cpu.set_weight(std::move(weight_cpu));
  ASSERT_TRUE(init_result);
  init_result = op_cpu.init();
  ASSERT_TRUE(init_result);

  MatMulOp op_cuda(input_dim, output_dim);
  op_cuda.set_device(DeviceType::CUDA);
  init_result = op_cuda.set_weight(std::move(weight_cuda));
  ASSERT_TRUE(init_result);
  init_result = op_cuda.init();
  ASSERT_TRUE(init_result);

  // Run forward
  auto forward_result = op_cpu.forward(input_cpu, output_cpu);
  ASSERT_TRUE(forward_result) << "CPU forward failed: " << forward_result.error().message();

  forward_result = op_cuda.forward(input_cuda, output_cuda);
  ASSERT_TRUE(forward_result) << "CUDA forward failed: " << forward_result.error().message();

  // Synchronize and compare
  err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess);

  auto output_cuda_cpu_result = Tensor::create({output_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cuda_cpu_result);
  auto output_cuda_cpu = std::move(output_cuda_cpu_result.value());
  err = cudaMemcpy(output_cuda_cpu.data(), output_cuda.data(),
                   output_dim * sizeof(f32), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Compare with tolerance
  const f32* cpu_data = output_cpu.ptr<f32>();
  const f32* cuda_data = output_cuda_cpu.ptr<f32>();
  for (i32 i = 0; i < output_dim; ++i) {
    EXPECT_NEAR(cpu_data[i], cuda_data[i], 1e-4f)
        << "Mismatch at index " << i;
  }

  std::cout << "✅ CUDA MatMul test passed!" << std::endl;
}

// Test CUDA RMSNorm
TEST(CUDAOpsTest, RMSNorm) {
  if (!is_cuda_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  constexpr i32 dim = 64;  // Must be multiple of 4
  constexpr f32 eps = 1e-5f;

  // Create weight tensors
  auto weight_cpu_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cpu_result);
  auto weight_cpu = std::move(weight_cpu_result.value());
  for (i32 i = 0; i < dim; ++i) {
    weight_cpu.ptr<f32>()[i] = 1.0f;  // Initialize to 1
  }

  auto weight_cuda_result = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(weight_cuda_result);
  auto weight_cuda = std::move(weight_cuda_result.value());
  cudaError_t err = cudaMemcpy(weight_cuda.data(), weight_cpu.data(),
                               dim * sizeof(f32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // Create input tensors
  auto input_cpu_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input_cpu_result);
  auto input_cpu = std::move(input_cpu_result.value());
  for (i32 i = 0; i < dim; ++i) {
    input_cpu.ptr<f32>()[i] = static_cast<f32>(i) * 0.1f;
  }

  auto input_cuda_result = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(input_cuda_result);
  auto input_cuda = std::move(input_cuda_result.value());
  err = cudaMemcpy(input_cuda.data(), input_cpu.data(),
                   dim * sizeof(f32), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // Create output tensors
  auto output_cpu_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cpu_result);
  auto output_cpu = std::move(output_cpu_result.value());

  auto output_cuda_result = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(output_cuda_result);
  auto output_cuda = std::move(output_cuda_result.value());

  // Create and initialize operators
  RMSNormOp op_cpu(dim, eps);
  op_cpu.set_device(DeviceType::CPU);
  auto init_result = op_cpu.set_weight(std::move(weight_cpu));
  ASSERT_TRUE(init_result);
  init_result = op_cpu.init();
  ASSERT_TRUE(init_result);

  RMSNormOp op_cuda(dim, eps);
  op_cuda.set_device(DeviceType::CUDA);
  init_result = op_cuda.set_weight(std::move(weight_cuda));
  ASSERT_TRUE(init_result);
  init_result = op_cuda.init();
  ASSERT_TRUE(init_result);

  // Run forward
  auto forward_result = op_cpu.forward(input_cpu, output_cpu);
  ASSERT_TRUE(forward_result) << "CPU forward failed: " << forward_result.error().message();

  forward_result = op_cuda.forward(input_cuda, output_cuda);
  ASSERT_TRUE(forward_result) << "CUDA forward failed: " << forward_result.error().message();

  // Synchronize and compare
  err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess);

  auto output_cuda_cpu_result = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_cuda_cpu_result);
  auto output_cuda_cpu = std::move(output_cuda_cpu_result.value());
  err = cudaMemcpy(output_cuda_cpu.data(), output_cuda.data(),
                   dim * sizeof(f32), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Compare with tolerance
  const f32* cpu_data = output_cpu.ptr<f32>();
  const f32* cuda_data = output_cuda_cpu.ptr<f32>();
  for (i32 i = 0; i < dim; ++i) {
    EXPECT_NEAR(cpu_data[i], cuda_data[i], 1e-4f)
        << "Mismatch at index " << i;
  }

  std::cout << "✅ CUDA RMSNorm test passed!" << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
