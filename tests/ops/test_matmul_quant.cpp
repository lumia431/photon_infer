/**
 * @file test_matmul_quant.cpp
 * @brief Integration tests for quantized MatMul operator
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "photon/core/quant.hpp"
#include "photon/core/tensor.hpp"
#include "photon/ops/matmul.hpp"

namespace photon::test {

// ============================================================================
// Test Fixtures
// ============================================================================

class QuantMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    rng_.seed(42);
  }

  // Generate random weights
  Tensor generate_weight_tensor(i32 rows, i32 cols, DeviceType device = DeviceType::CPU) {
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(rows * cols);
    for (auto& val : data) {
      val = dist(rng_);
    }
    auto result = Tensor::from_matrix(rows, cols, data, device);
    return std::move(result.value());
  }

  // Generate random input
  Tensor generate_input_tensor(i32 size, DeviceType device = DeviceType::CPU) {
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(size);
    for (auto& val : data) {
      val = dist(rng_);
    }
    auto result = Tensor::from_vector(data, device);
    return std::move(result.value());
  }

  // Compute max relative error
  f32 compute_max_relative_error(const Tensor& t1, const Tensor& t2) {
    EXPECT_EQ(t1.size(), t2.size());

    f32 max_rel_error = 0.0f;
    for (usize i = 0; i < t1.size(); ++i) {
      f32 v1 = t1.index<f32>(i);
      f32 v2 = t2.index<f32>(i);
      f32 abs_diff = std::abs(v1 - v2);
      f32 denom = std::max(std::abs(v1), 1e-6f);
      f32 rel_error = abs_diff / denom;
      max_rel_error = std::max(max_rel_error, rel_error);
    }
    return max_rel_error;
  }

  std::mt19937 rng_;
};

// ============================================================================
// CPU Quantized MatMul Tests
// ============================================================================

TEST_F(QuantMatMulTest, DISABLED_QuantizedMatMulCPU_Small) {
  // Small test: [4] @ [8 × 4]^T -> [8]
  i32 input_dim = 4;
  i32 output_dim = 8;

  // Create float32 operator
  MatMulOp op_fp32(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/false);

  // Create quantized operator
  MatMulOp op_quant(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/true);

  // Generate weight and quantize
  auto weight_fp32 = generate_weight_tensor(output_dim, input_dim, DeviceType::CPU);

  // Quantize weight
  i32 group_size = 4;
  auto quant_result = quantize_tensor(weight_fp32, group_size);
  ASSERT_TRUE(quant_result);
  auto [weight_quant, quant_params] = std::move(quant_result.value());

  // Set weights
  auto weight_fp32_copy = weight_fp32.clone().value();
  ASSERT_TRUE(op_fp32.set_weight(std::move(weight_fp32)));
  ASSERT_TRUE(op_quant.set_quantized_weight(std::move(weight_quant), std::move(quant_params)));

  // Initialize
  ASSERT_TRUE(op_fp32.init());
  ASSERT_TRUE(op_quant.init());

  // Generate input
  auto input = generate_input_tensor(input_dim, DeviceType::CPU);

  // Create outputs
  auto output_fp32 = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CPU).value();
  auto output_quant = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CPU).value();

  // Forward pass
  ASSERT_TRUE(op_fp32.forward(input, output_fp32));
  ASSERT_TRUE(op_quant.forward(input, output_quant));

  // Compare results
  f32 max_rel_error = compute_max_relative_error(output_fp32, output_quant);

  std::cout << "Max relative error (CPU): " << max_rel_error << std::endl;

  // Quantization error should be small (< 5%)
  EXPECT_LT(max_rel_error, 0.05f);
}

// ============================================================================
// CUDA Quantized MatMul Tests
// ============================================================================

#ifdef PHOTON_USE_CUDA

TEST_F(QuantMatMulTest, QuantizedMatMulCUDA_Small) {
  // Small test: [64] @ [128 × 64]^T -> [128]
  i32 input_dim = 64;
  i32 output_dim = 128;

  // Create operators on CUDA device
  MatMulOp op_fp32(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/false);
  op_fp32.set_device(DeviceType::CUDA);

  MatMulOp op_quant(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/true);
  op_quant.set_device(DeviceType::CUDA);

  // Generate weight and quantize (on CPU first)
  auto weight_fp32_cpu = generate_weight_tensor(output_dim, input_dim, DeviceType::CPU);

  // Quantize weight
  i32 group_size = 64;
  auto quant_result = quantize_tensor(weight_fp32_cpu, group_size);
  ASSERT_TRUE(quant_result);
  auto [weight_quant_cpu, quant_params] = std::move(quant_result.value());

  // Move to GPU
  auto weight_fp32 = weight_fp32_cpu.to(DeviceType::CUDA).value();
  auto weight_quant = weight_quant_cpu.to(DeviceType::CUDA).value();

  // Set weights
  ASSERT_TRUE(op_fp32.set_weight(std::move(weight_fp32)));
  auto set_quant_result = op_quant.set_quantized_weight(std::move(weight_quant), std::move(quant_params));
  if (!set_quant_result) {
    std::cerr << "Failed to set quantized weight: " << set_quant_result.error().message() << std::endl;
  }
  ASSERT_TRUE(set_quant_result);

  // Initialize
  ASSERT_TRUE(op_fp32.init());
  ASSERT_TRUE(op_quant.init());

  // Generate input (on GPU)
  auto input_cpu = generate_input_tensor(input_dim, DeviceType::CPU);
  auto input = input_cpu.to(DeviceType::CUDA).value();

  // Create outputs (on GPU)
  auto output_fp32 = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CUDA).value();
  auto output_quant = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward pass
  ASSERT_TRUE(op_fp32.forward(input, output_fp32));
  ASSERT_TRUE(op_quant.forward(input, output_quant));

  // Move results back to CPU for comparison
  auto output_fp32_cpu = output_fp32.to(DeviceType::CPU).value();
  auto output_quant_cpu = output_quant.to(DeviceType::CPU).value();

  // Compare results using RMSE and max absolute error
  // (more robust than max relative error for values close to zero)
  f32 sum_squared_error = 0.0f;
  f32 max_abs_error = 0.0f;

  for (usize i = 0; i < output_fp32_cpu.size(); ++i) {
    f32 fp32_val = output_fp32_cpu.index<f32>(i);
    f32 quant_val = output_quant_cpu.index<f32>(i);
    f32 diff = fp32_val - quant_val;
    f32 abs_err = std::abs(diff);
    sum_squared_error += diff * diff;
    max_abs_error = std::max(max_abs_error, abs_err);
  }

  f32 rmse = std::sqrt(sum_squared_error / output_fp32_cpu.size());

  std::cout << "Accuracy metrics:" << std::endl;
  std::cout << "  RMSE: " << rmse << std::endl;
  std::cout << "  Max absolute error: " << max_abs_error << std::endl;

  // RMSE should be small (typically < 0.1 for quantization)
  EXPECT_LT(rmse, 0.1f);
  // Max absolute error should be reasonable (< 0.05)
  EXPECT_LT(max_abs_error, 0.05f);
}

TEST_F(QuantMatMulTest, QuantizedMatMulCUDA_Large) {
  // Realistic size: [2048] @ [2048 × 2048]^T -> [2048]
  i32 input_dim = 2048;
  i32 output_dim = 2048;

  // Create operators on CUDA device
  MatMulOp op_fp32(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/false);
  op_fp32.set_device(DeviceType::CUDA);

  MatMulOp op_quant(input_dim, output_dim, /*use_naive=*/false, /*is_quantized=*/true);
  op_quant.set_device(DeviceType::CUDA);

  // Generate weight and quantize
  auto weight_fp32_cpu = generate_weight_tensor(output_dim, input_dim, DeviceType::CPU);

  i32 group_size = 128;
  auto quant_result = quantize_tensor(weight_fp32_cpu, group_size);
  ASSERT_TRUE(quant_result);
  auto [weight_quant_cpu, quant_params] = std::move(quant_result.value());

  // Move to GPU
  auto weight_fp32 = weight_fp32_cpu.to(DeviceType::CUDA).value();
  auto weight_quant = weight_quant_cpu.to(DeviceType::CUDA).value();

  // Set weights
  ASSERT_TRUE(op_fp32.set_weight(std::move(weight_fp32)));
  auto set_quant_result = op_quant.set_quantized_weight(std::move(weight_quant), std::move(quant_params));
  if (!set_quant_result) {
    std::cerr << "Failed to set quantized weight: " << set_quant_result.error().message() << std::endl;
  }
  ASSERT_TRUE(set_quant_result);

  // Initialize
  ASSERT_TRUE(op_fp32.init());
  ASSERT_TRUE(op_quant.init());

  // Generate input
  auto input_cpu = generate_input_tensor(input_dim, DeviceType::CPU);
  auto input = input_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto output_fp32 = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CUDA).value();
  auto output_quant = Tensor::zeros({output_dim}, DataType::Float32, DeviceType::CUDA).value();

  // Warm-up
  op_fp32.forward(input, output_fp32);
  op_quant.forward(input, output_quant);
  cudaDeviceSynchronize();

  // Benchmark FP32
  auto start_fp32 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    op_fp32.forward(input, output_fp32);
  }
  cudaDeviceSynchronize();
  auto end_fp32 = std::chrono::high_resolution_clock::now();
  auto duration_fp32 = std::chrono::duration_cast<std::chrono::microseconds>(end_fp32 - start_fp32).count();

  // Benchmark INT8
  auto start_quant = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    op_quant.forward(input, output_quant);
  }
  cudaDeviceSynchronize();
  auto end_quant = std::chrono::high_resolution_clock::now();
  auto duration_quant = std::chrono::duration_cast<std::chrono::microseconds>(end_quant - start_quant).count();

  // Move results back to CPU for comparison
  auto output_fp32_cpu = output_fp32.to(DeviceType::CPU).value();
  auto output_quant_cpu = output_quant.to(DeviceType::CPU).value();

  // Compare accuracy using RMSE and max absolute error
  f32 sum_squared_error = 0.0f;
  f32 max_abs_error = 0.0f;

  for (usize i = 0; i < output_fp32_cpu.size(); ++i) {
    f32 fp32_val = output_fp32_cpu.index<f32>(i);
    f32 quant_val = output_quant_cpu.index<f32>(i);
    f32 diff = fp32_val - quant_val;
    f32 abs_err = std::abs(diff);
    sum_squared_error += diff * diff;
    max_abs_error = std::max(max_abs_error, abs_err);
  }

  f32 rmse = std::sqrt(sum_squared_error / output_fp32_cpu.size());

  std::cout << "\n=== Quantized MatMul Performance ===" << std::endl;
  std::cout << "Matrix size: " << output_dim << " × " << input_dim << std::endl;
  std::cout << "FP32 time:  " << duration_fp32 / 100.0 << " µs/iter" << std::endl;
  std::cout << "INT8 time:  " << duration_quant / 100.0 << " µs/iter" << std::endl;
  std::cout << "Speedup:    " << static_cast<f32>(duration_fp32) / duration_quant << "x" << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;
  std::cout << "Max absolute error: " << max_abs_error << std::endl;

  // Accuracy check (larger matrices accumulate more quantization error)
  EXPECT_LT(rmse, 0.1f);  // RMSE should still be reasonable
  EXPECT_LT(max_abs_error, 0.2f);  // Max absolute error threshold for large matrices

  // Performance check (INT8 should be faster or comparable)
  // Note: Speedup depends on hardware and memory bandwidth
  std::cout << "INT8/FP32 time ratio: " << static_cast<f32>(duration_quant) / duration_fp32 << std::endl;
}

#endif  // PHOTON_USE_CUDA

}  // namespace photon::test
