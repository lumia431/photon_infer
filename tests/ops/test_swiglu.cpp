/**
 * @file test_swiglu.cpp
 * @brief Unit tests for SwiGLU operator
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <vector>
#include "photon/ops/swiglu.hpp"

using namespace photon;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(SwiGLUOpTest, BasicConstruction) {
  const i32 hidden_dim = 512;

  SwiGLUOp op(hidden_dim);

  EXPECT_EQ(op.hidden_dim(), hidden_dim);
  EXPECT_FALSE(op.is_initialized());
  EXPECT_EQ(op.device(), DeviceType::CPU);
  EXPECT_FALSE(op.is_naive());
}

TEST(SwiGLUOpTest, Initialization) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);

  // Initialize
  auto init_result = op.init();
  ASSERT_TRUE(init_result);
  EXPECT_TRUE(op.is_initialized());
}

TEST(SwiGLUOpTest, UninitializedForward) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);

  auto input1 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto input2 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  // Forward without initialization should fail
  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidOperator);
}

// ============================================================================
// Input Validation Tests
// ============================================================================

TEST(SwiGLUOpTest, InvalidInput1Shape) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  // Wrong shape: 2D instead of 1D
  auto input1 = Tensor::create({2, 2}, DataType::Float32).value();
  auto input2 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(SwiGLUOpTest, InvalidInput2Shape) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({hidden_dim}, DataType::Float32).value();
  // Wrong shape: 2D instead of 1D
  auto input2 = Tensor::create({2, 2}, DataType::Float32).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(SwiGLUOpTest, InvalidOutputShape) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto input2 = Tensor::create({hidden_dim}, DataType::Float32).value();
  // Wrong shape: 2D instead of 1D
  auto output = Tensor::create({2, 2}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(SwiGLUOpTest, InvalidInput1Dimension) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  // Wrong dimension: [8] instead of [4]
  auto input1 = Tensor::create({8}, DataType::Float32).value();
  auto input2 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(SwiGLUOpTest, InvalidInput2Dimension) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({hidden_dim}, DataType::Float32).value();
  // Wrong dimension: [8] instead of [4]
  auto input2 = Tensor::create({8}, DataType::Float32).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(SwiGLUOpTest, InvalidOutputDimension) {
  const i32 hidden_dim = 4;

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({hidden_dim}, DataType::Float32).value();
  auto input2 = Tensor::create({hidden_dim}, DataType::Float32).value();
  // Wrong dimension: [8] instead of [4]
  auto output = Tensor::create({8}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// Functional Tests
// ============================================================================

TEST(SwiGLUOpTest, SimpleSwiGLU) {
  const i32 hidden_dim = 4;

  // Input1 (gate): [1, 2, 3, 4]
  std::vector<f32> input1_data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Input2 (value): [0.5, 1.0, 1.5, 2.0]
  std::vector<f32> input2_data = {0.5f, 1.0f, 1.5f, 2.0f};

  // Create operator
  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::from_vector(input1_data).value();
  auto input2 = Tensor::from_vector(input2_data).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input1, input2, output);
  ASSERT_TRUE(result);

  // Calculate expected output manually
  // SiLU(x) = x / (1 + exp(-x))
  // output[i] = SiLU(input1[i]) * input2[i]
  auto out_vec = output.vector_map<f32>();

  for (i32 i = 0; i < hidden_dim; ++i) {
    f32 x = input1_data[i];
    f32 silu = x / (1.0f + std::exp(-x));
    f32 expected = silu * input2_data[i];
    EXPECT_NEAR(out_vec(i), expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST(SwiGLUOpTest, ZeroInput) {
  const i32 hidden_dim = 4;

  // All zeros
  std::vector<f32> input1_data(hidden_dim, 0.0f);
  std::vector<f32> input2_data(hidden_dim, 1.0f);

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::from_vector(input1_data).value();
  auto input2 = Tensor::from_vector(input2_data).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  ASSERT_TRUE(result);

  // SiLU(0) = 0 / (1 + exp(0)) = 0 / (1 + 1) = 0
  // output = 0 * input2 = 0
  auto out_vec = output.vector_map<f32>();
  for (i32 i = 0; i < hidden_dim; ++i) {
    EXPECT_NEAR(out_vec(i), 0.0f, 1e-6f);
  }
}

TEST(SwiGLUOpTest, LargeDimension) {
  const i32 hidden_dim = 256;

  // Create random-like inputs
  std::vector<f32> input1_data(hidden_dim);
  std::vector<f32> input2_data(hidden_dim);
  for (i32 i = 0; i < hidden_dim; ++i) {
    input1_data[i] = std::sin(static_cast<f32>(i) * 0.1f);
    input2_data[i] = std::cos(static_cast<f32>(i) * 0.05f);
  }

  SwiGLUOp op(hidden_dim);
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::from_vector(input1_data).value();
  auto input2 = Tensor::from_vector(input2_data).value();
  auto output = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result = op.forward(input1, input2, output);
  ASSERT_TRUE(result);

  // Verify computation is correct
  auto out_vec = output.vector_map<f32>();
  for (i32 i = 0; i < hidden_dim; ++i) {
    f32 x = input1_data[i];
    f32 silu = x / (1.0f + std::exp(-x));
    f32 expected = silu * input2_data[i];
    EXPECT_NEAR(out_vec(i), expected, 1e-4f) << "Mismatch at index " << i;
  }
}

// ============================================================================
// Correctness: Naive vs Eigen Implementation
// ============================================================================

TEST(SwiGLUOpTest, NaiveVsEigen) {
  const i32 hidden_dim = 128;

  // Create test inputs
  std::vector<f32> input1_data(hidden_dim);
  std::vector<f32> input2_data(hidden_dim);
  for (i32 i = 0; i < hidden_dim; ++i) {
    input1_data[i] = static_cast<f32>(i - hidden_dim / 2) * 0.1f;  // -6.4 to 6.3
    input2_data[i] = std::sin(static_cast<f32>(i) * 0.05f);
  }

  // Naive implementation
  SwiGLUOp op_naive(hidden_dim, true);  // use_naive = true
  ASSERT_TRUE(op_naive.init());

  auto input1_naive = Tensor::from_vector(input1_data).value();
  auto input2_naive = Tensor::from_vector(input2_data).value();
  auto output_naive = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result_naive = op_naive.forward(input1_naive, input2_naive, output_naive);
  ASSERT_TRUE(result_naive);

  // Eigen implementation
  SwiGLUOp op_eigen(hidden_dim, false);  // use_naive = false
  ASSERT_TRUE(op_eigen.init());

  auto input1_eigen = Tensor::from_vector(input1_data).value();
  auto input2_eigen = Tensor::from_vector(input2_data).value();
  auto output_eigen = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto result_eigen = op_eigen.forward(input1_eigen, input2_eigen, output_eigen);
  ASSERT_TRUE(result_eigen);

  // Compare outputs
  auto out_naive = output_naive.vector_map<f32>();
  auto out_eigen = output_eigen.vector_map<f32>();

  for (i32 i = 0; i < hidden_dim; ++i) {
    EXPECT_NEAR(out_naive(i), out_eigen(i), 1e-5f) << "Mismatch at index " << i;
  }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST(SwiGLUOpTest, Benchmark) {
  const i32 hidden_dim = 512;
  const i32 iterations = 10000;

  // Create test inputs
  std::vector<f32> input1_data(hidden_dim);
  std::vector<f32> input2_data(hidden_dim);
  for (i32 i = 0; i < hidden_dim; ++i) {
    input1_data[i] = static_cast<f32>(i) * 0.01f - 2.56f;  // -2.56 to 2.55
    input2_data[i] = std::sin(static_cast<f32>(i) * 0.02f);
  }

  // Naive benchmark
  SwiGLUOp op_naive(hidden_dim, true);
  ASSERT_TRUE(op_naive.init());

  auto input1_naive = Tensor::from_vector(input1_data).value();
  auto input2_naive = Tensor::from_vector(input2_data).value();
  auto output_naive = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_naive.forward(input1_naive, input2_naive, output_naive);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Eigen benchmark
  SwiGLUOp op_eigen(hidden_dim, false);
  ASSERT_TRUE(op_eigen.init());

  auto input1_eigen = Tensor::from_vector(input1_data).value();
  auto input2_eigen = Tensor::from_vector(input2_data).value();
  auto output_eigen = Tensor::create({hidden_dim}, DataType::Float32).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_eigen.forward(input1_eigen, input2_eigen, output_eigen);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\nSwiGLU Benchmark (dim=" << hidden_dim << ", "
            << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << (duration_naive / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << (duration_eigen / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Speedup: " << (static_cast<double>(duration_naive) / duration_eigen)
            << "x\n";

  // Eigen should be faster
  EXPECT_LT(duration_eigen, duration_naive);
}

