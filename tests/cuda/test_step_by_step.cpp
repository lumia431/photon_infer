/**
 * @file test_step_by_step.cpp
 * @brief Step-by-step CUDA operator validation
 *
 * This test validates each operator by comparing CPU vs CUDA outputs
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cmath>

#include "photon/core/tensor.hpp"
#include "photon/ops/embedding.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/rope.hpp"
#include "photon/ops/add.hpp"
#include "photon/ops/swiglu.hpp"
#include "photon/model/checkpoint.hpp"
#include "photon/model/llama_model.hpp"

using namespace photon;
using namespace photon::model;

// Helper: compare two tensors with tolerance
bool tensors_close(const Tensor& a, const Tensor& b, f32 rtol = 1e-4f, f32 atol = 1e-5f) {
  if (a.size() != b.size()) return false;

  // Convert to CPU if needed (to() returns a copy, safe even if already on CPU)
  auto a_cpu_result = a.to(DeviceType::CPU);
  auto b_cpu_result = b.to(DeviceType::CPU);

  if (!a_cpu_result || !b_cpu_result) {
    std::cout << "Failed to convert tensors to CPU\n";
    return false;
  }

  Tensor a_cpu = std::move(a_cpu_result.value());
  Tensor b_cpu = std::move(b_cpu_result.value());

  auto a_map = a_cpu.vector_map<f32>();
  auto b_map = b_cpu.vector_map<f32>();

  for (usize i = 0; i < a.size(); ++i) {
    f32 diff = std::abs(a_map[i] - b_map[i]);
    f32 threshold = atol + rtol * std::abs(b_map[i]);
    if (diff > threshold) {
      std::cout << "Mismatch at index " << i << ": "
                << a_map[i] << " vs " << b_map[i]
                << " (diff=" << diff << ", threshold=" << threshold << ")\n";
      return false;
    }
  }
  return true;
}

class StepByStepTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Small test dimensions
    vocab_size_ = 100;
    dim_ = 64;
    hidden_dim_ = 256;
    head_num_ = 8;
    kv_head_num_ = 2;
    head_size_ = 8;
    seq_len_ = 128;
  }

  i32 vocab_size_;
  i32 dim_;
  i32 hidden_dim_;
  i32 head_num_;
  i32 kv_head_num_;
  i32 head_size_;
  i32 seq_len_;
};

TEST_F(StepByStepTest, Step1_Embedding) {
  std::cout << "\n=== Step 1: Testing Embedding ===\n";

  // Create operators
  EmbeddingOp emb_cpu(vocab_size_, dim_);
  EmbeddingOp emb_cuda(vocab_size_, dim_);
  emb_cuda.set_device(DeviceType::CUDA);

  // Create weight on CPU
  auto weight_result = Tensor::zeros({vocab_size_, dim_}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_result);

  auto weight_map = weight_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size_; ++i) {
    for (i32 j = 0; j < dim_; ++j) {
      weight_map(i, j) = 0.01f * static_cast<f32>(i + j);
    }
  }

  // Create separate weight tensors for CPU and CUDA (can't copy, need to create independently)
  auto weight_cpu_result = Tensor::zeros({vocab_size_, dim_}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cpu_result);
  auto cpu_map = weight_cpu_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size_; ++i) {
    for (i32 j = 0; j < dim_; ++j) {
      cpu_map(i, j) = 0.01f * static_cast<f32>(i + j);
    }
  }

  auto weight_cuda_result = Tensor::zeros({vocab_size_, dim_}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cuda_result);
  auto cuda_map = weight_cuda_result.value().matrix_map<f32>();
  for (i32 i = 0; i < vocab_size_; ++i) {
    for (i32 j = 0; j < dim_; ++j) {
      cuda_map(i, j) = 0.01f * static_cast<f32>(i + j);
    }
  }

  // Set weights
  ASSERT_TRUE(emb_cpu.set_weight(std::move(weight_cpu_result.value())));
  ASSERT_TRUE(emb_cpu.init());

  ASSERT_TRUE(emb_cuda.set_weight(std::move(weight_cuda_result.value())));
  ASSERT_TRUE(emb_cuda.init());

  // Create input tokens
  auto tokens_cpu = Tensor::create({3}, DataType::Int32, DeviceType::CPU).value();
  auto tokens_cuda = Tensor::create({3}, DataType::Int32, DeviceType::CUDA).value();

  // Set token values: [1, 5, 10]
  i32 token_vals[3] = {1, 5, 10};
  auto tokens_map = tokens_cpu.vector_map<i32>();
  for (int i = 0; i < 3; ++i) {
    tokens_map[i] = token_vals[i];
  }

  // Copy to CUDA
  cudaMemcpy(tokens_cuda.data(), tokens_cpu.data(), 3 * sizeof(i32), cudaMemcpyHostToDevice);

  // Create outputs
  auto out_cpu = Tensor::create({3, dim_}, DataType::Float32, DeviceType::CPU).value();
  auto out_cuda = Tensor::create({3, dim_}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "Running CPU embedding...\n";
  ASSERT_TRUE(emb_cpu.forward(tokens_cpu, out_cpu));

  std::cout << "Running CUDA embedding...\n";
  ASSERT_TRUE(emb_cuda.forward(tokens_cuda, out_cuda));

  // Compare
  std::cout << "Comparing outputs...\n";
  EXPECT_TRUE(tensors_close(out_cpu, out_cuda));

  // Print first few values
  auto cpu_out_map = out_cpu.matrix_map<f32>();
  auto cuda_result = out_cuda.to(DeviceType::CPU).value();
  auto cuda_out_map = cuda_result.matrix_map<f32>();

  std::cout << "First token embedding (first 5 dims):\n";
  std::cout << "  CPU:  ";
  for (int i = 0; i < 5; ++i) {
    std::cout << cpu_out_map(0, i) << " ";
  }
  std::cout << "\n  CUDA: ";
  for (int i = 0; i < 5; ++i) {
    std::cout << cuda_out_map(0, i) << " ";
  }
  std::cout << "\n✅ Embedding test passed!\n";
}

TEST_F(StepByStepTest, Step2_MatMul) {
  std::cout << "\n=== Step 2: Testing MatMul (QKV projection) ===\n";

  // Create operators
  MatMulOp matmul_cpu(dim_, dim_);
  MatMulOp matmul_cuda(dim_, dim_);
  matmul_cuda.set_device(DeviceType::CUDA);

  // Create separate weight tensors for CPU and CUDA
  auto weight_cpu_result = Tensor::zeros({dim_, dim_}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cpu_result);
  auto cpu_map = weight_cpu_result.value().matrix_map<f32>();
  for (i32 i = 0; i < dim_; ++i) {
    for (i32 j = 0; j < dim_; ++j) {
      cpu_map(i, j) = 0.01f * static_cast<f32>(i - j);
    }
  }

  auto weight_cuda_result = Tensor::zeros({dim_, dim_}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(weight_cuda_result);
  auto cuda_map = weight_cuda_result.value().matrix_map<f32>();
  for (i32 i = 0; i < dim_; ++i) {
    for (i32 j = 0; j < dim_; ++j) {
      cuda_map(i, j) = 0.01f * static_cast<f32>(i - j);
    }
  }

  // Set weights
  ASSERT_TRUE(matmul_cpu.set_weight(std::move(weight_cpu_result.value())));
  ASSERT_TRUE(matmul_cpu.init());

  ASSERT_TRUE(matmul_cuda.set_weight(std::move(weight_cuda_result.value())));
  ASSERT_TRUE(matmul_cuda.init());

  // Create input
  auto input_cpu = Tensor::zeros({dim_}, DataType::Float32, DeviceType::CPU).value();
  auto input_map = input_cpu.vector_map<f32>();
  for (i32 i = 0; i < dim_; ++i) {
    input_map[i] = 0.1f * static_cast<f32>(i);
  }

  auto input_cuda = input_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto out_cpu = Tensor::create({dim_}, DataType::Float32, DeviceType::CPU).value();
  auto out_cuda = Tensor::create({dim_}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "Running CPU matmul...\n";
  ASSERT_TRUE(matmul_cpu.forward(input_cpu, out_cpu));

  std::cout << "Running CUDA matmul...\n";
  ASSERT_TRUE(matmul_cuda.forward(input_cuda, out_cuda));

  // Compare
  std::cout << "Comparing outputs...\n";
  EXPECT_TRUE(tensors_close(out_cpu, out_cuda));

  std::cout << "✅ MatMul test passed!\n";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
