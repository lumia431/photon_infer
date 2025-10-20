/**
 * @file test_rmsnorm.cpp
 * @brief Unit tests for RMSNorm operator
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <vector>
#include "photon/ops/rmsnorm.hpp"

using namespace photon;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(RMSNormOpTest, BasicConstruction) {
  const i32 dim = 512;
  const f32 eps = 1e-5f;

  RMSNormOp op(dim, eps);

  EXPECT_EQ(op.dim(), dim);
  EXPECT_FLOAT_EQ(op.eps(), eps);
  EXPECT_FALSE(op.is_initialized());
  EXPECT_EQ(op.device(), DeviceType::CPU);
  EXPECT_FALSE(op.is_naive());
}

TEST(RMSNormOpTest, WeightSetting) {
  const i32 dim = 4;

  RMSNormOp op(dim);

  // Create weight tensor [dim]
  auto weight_result = Tensor::create({dim}, DataType::Float32);
  ASSERT_TRUE(weight_result);

  auto weight = std::move(weight_result.value());

  // Fill with sequential values
  f32* data_ptr = weight.ptr<f32>();
  for (usize i = 0; i < weight.size(); ++i) {
    data_ptr[i] = static_cast<f32>(i + 1);
  }

  // Set weight
  auto set_result = op.set_weight(std::move(weight));
  ASSERT_TRUE(set_result);

  // Initialize
  auto init_result = op.init();
  ASSERT_TRUE(init_result);
  EXPECT_TRUE(op.is_initialized());
}

TEST(RMSNormOpTest, InvalidWeightShape) {
  const i32 dim = 4;

  RMSNormOp op(dim);

  // Wrong shape: 2D instead of 1D
  auto weight = Tensor::create({2, 2}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(RMSNormOpTest, InvalidWeightDimension) {
  const i32 dim = 4;

  RMSNormOp op(dim);

  // Wrong dimension: [8] instead of [4]
  auto weight = Tensor::create({8}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(RMSNormOpTest, UninitializedForward) {
  const i32 dim = 4;

  RMSNormOp op(dim);

  auto input = Tensor::create({dim}, DataType::Float32).value();
  auto output = Tensor::create({dim}, DataType::Float32).value();

  // Forward without initialization should fail
  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidOperator);
}

// ============================================================================
// Single Vector Normalization Tests
// ============================================================================

TEST(RMSNormOpTest, SimpleNormalization) {
  const i32 dim = 4;

  // Input: [1, 2, 3, 4]
  std::vector<f32> input_data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Weight: [1, 1, 1, 1] (no scaling)
  std::vector<f32> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

  // Create operator
  RMSNormOp op(dim, 1e-5f);
  auto weight = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input and output tensors
  auto input = Tensor::from_vector(input_data).value();
  auto output = Tensor::create({dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result) << result.error().to_string();

  // Calculate expected output manually
  // mean_sq = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
  // rms = sqrt(7.5 + 1e-5) ≈ 2.7386
  // rsqrt = 1 / 2.7386 ≈ 0.3651
  // output = input * rsqrt
  f32 mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
  f32 rsqrt = 1.0f / std::sqrt(mean_sq + 1e-5f);

  auto out_vec = output.vector_map<f32>();
  EXPECT_NEAR(out_vec(0), 1.0f * rsqrt, 1e-5f);
  EXPECT_NEAR(out_vec(1), 2.0f * rsqrt, 1e-5f);
  EXPECT_NEAR(out_vec(2), 3.0f * rsqrt, 1e-5f);
  EXPECT_NEAR(out_vec(3), 4.0f * rsqrt, 1e-5f);

  // Verify normalization property: mean of squares should be close to 1
  f32 output_mean_sq = 0.0f;
  for (i32 i = 0; i < dim; ++i) {
    output_mean_sq += out_vec(i) * out_vec(i);
  }
  output_mean_sq /= static_cast<f32>(dim);
  EXPECT_NEAR(output_mean_sq, 1.0f, 1e-4f);
}

TEST(RMSNormOpTest, NormalizationWithScaling) {
  const i32 dim = 4;

  // Input: [2, 4, 6, 8]
  std::vector<f32> input_data = {2.0f, 4.0f, 6.0f, 8.0f};

  // Weight: [0.5, 1.0, 1.5, 2.0] (different scales)
  std::vector<f32> weight_data = {0.5f, 1.0f, 1.5f, 2.0f};

  // Create operator
  RMSNormOp op(dim, 1e-5f);
  auto weight = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_vector(input_data).value();
  auto output = Tensor::create({dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Calculate expected output
  // mean_sq = (4 + 16 + 36 + 64) / 4 = 120 / 4 = 30
  // rsqrt = 1 / sqrt(30 + 1e-5) ≈ 0.1826
  f32 mean_sq = (4.0f + 16.0f + 36.0f + 64.0f) / 4.0f;
  f32 rsqrt = 1.0f / std::sqrt(mean_sq + 1e-5f);

  auto out_vec = output.vector_map<f32>();
  EXPECT_NEAR(out_vec(0), 2.0f * rsqrt * 0.5f, 1e-5f);
  EXPECT_NEAR(out_vec(1), 4.0f * rsqrt * 1.0f, 1e-5f);
  EXPECT_NEAR(out_vec(2), 6.0f * rsqrt * 1.5f, 1e-5f);
  EXPECT_NEAR(out_vec(3), 8.0f * rsqrt * 2.0f, 1e-5f);
}

TEST(RMSNormOpTest, LargerDimension) {
  const i32 dim = 256;

  // Create input and weight
  std::vector<f32> input_data(dim);
  std::vector<f32> weight_data(dim);
  for (i32 i = 0; i < dim; ++i) {
    input_data[i] = static_cast<f32>(i) * 0.01f;
    weight_data[i] = 1.0f;
  }

  RMSNormOp op(dim, 1e-5f);
  auto weight = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_vector(input_data).value();
  auto output = Tensor::create({dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Verify normalization property
  f32 output_mean_sq = 0.0f;
  auto out_vec = output.vector_map<f32>();
  for (i32 i = 0; i < dim; ++i) {
    output_mean_sq += out_vec(i) * out_vec(i);
  }
  output_mean_sq /= static_cast<f32>(dim);
  EXPECT_NEAR(output_mean_sq, 1.0f, 1e-4f);
}

// ============================================================================
// Batch Normalization Tests
// ============================================================================

TEST(RMSNormOpTest, SimpleBatchNormalization) {
  const i32 batch_size = 2;
  const i32 dim = 4;

  // Input: [[1, 2, 3, 4],
  //         [5, 6, 7, 8]]
  std::vector<f32> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  // Weight: [1, 1, 1, 1]
  std::vector<f32> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

  // Create operator
  RMSNormOp op(dim, 1e-5f);
  auto weight = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output = Tensor::create({batch_size, dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Each batch item should be normalized independently
  auto out_mat = output.matrix_map<f32>();

  // Batch 0: mean_sq = (1 + 4 + 9 + 16) / 4 = 7.5
  f32 mean_sq_0 = 7.5f;
  f32 rsqrt_0 = 1.0f / std::sqrt(mean_sq_0 + 1e-5f);
  EXPECT_NEAR(out_mat(0, 0), 1.0f * rsqrt_0, 1e-5f);
  EXPECT_NEAR(out_mat(0, 1), 2.0f * rsqrt_0, 1e-5f);

  // Batch 1: mean_sq = (25 + 36 + 49 + 64) / 4 = 43.5
  f32 mean_sq_1 = 43.5f;
  f32 rsqrt_1 = 1.0f / std::sqrt(mean_sq_1 + 1e-5f);
  EXPECT_NEAR(out_mat(1, 0), 5.0f * rsqrt_1, 1e-5f);
  EXPECT_NEAR(out_mat(1, 1), 6.0f * rsqrt_1, 1e-5f);
}

TEST(RMSNormOpTest, LargerBatch) {
  const i32 batch_size = 8;
  const i32 dim = 128;

  // Create input and weight
  std::vector<f32> input_data(batch_size * dim);
  for (i32 b = 0; b < batch_size; ++b) {
    for (i32 d = 0; d < dim; ++d) {
      input_data[b * dim + d] = static_cast<f32>(b * dim + d) * 0.01f;
    }
  }

  std::vector<f32> weight_data(dim, 1.0f);

  RMSNormOp op(dim, 1e-5f);
  auto weight = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output = Tensor::create({batch_size, dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Verify each batch item is normalized
  auto out_mat = output.matrix_map<f32>();
  for (i32 b = 0; b < batch_size; ++b) {
    f32 batch_mean_sq = 0.0f;
    for (i32 d = 0; d < dim; ++d) {
      batch_mean_sq += out_mat(b, d) * out_mat(b, d);
    }
    batch_mean_sq /= static_cast<f32>(dim);
    EXPECT_NEAR(batch_mean_sq, 1.0f, 1e-4f);
  }
}

// ============================================================================
// Correctness: Naive vs Eigen Implementation
// ============================================================================

TEST(RMSNormOpTest, NaiveVsEigenSingleVector) {
  const i32 dim = 512;

  // Create input and weight
  std::vector<f32> input_data(dim);
  std::vector<f32> weight_data(dim);
  for (i32 i = 0; i < dim; ++i) {
    input_data[i] = std::sin(static_cast<f32>(i) * 0.1f);
    weight_data[i] = std::cos(static_cast<f32>(i) * 0.05f);
  }

  // Naive implementation
  RMSNormOp op_naive(dim, 1e-5f, true);  // use_naive = true
  auto weight1 = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op_naive.set_weight(std::move(weight1)));
  ASSERT_TRUE(op_naive.init());

  auto input1 = Tensor::from_vector(input_data).value();
  auto output1 = Tensor::create({dim}, DataType::Float32).value();
  auto result1 = op_naive.forward(input1, output1);
  ASSERT_TRUE(result1);

  // Eigen implementation
  RMSNormOp op_eigen(dim, 1e-5f, false);  // use_naive = false
  auto weight2 = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op_eigen.set_weight(std::move(weight2)));
  ASSERT_TRUE(op_eigen.init());

  auto input2 = Tensor::from_vector(input_data).value();
  auto output2 = Tensor::create({dim}, DataType::Float32).value();
  auto result2 = op_eigen.forward(input2, output2);
  ASSERT_TRUE(result2);

  // Compare outputs
  auto out1 = output1.vector_map<f32>();
  auto out2 = output2.vector_map<f32>();

  for (i32 i = 0; i < dim; ++i) {
    EXPECT_NEAR(out1(i), out2(i), 1e-5f) << "Mismatch at index " << i;
  }
}

TEST(RMSNormOpTest, NaiveVsEigenBatch) {
  const i32 batch_size = 16;
  const i32 dim = 256;

  // Create input and weight
  std::vector<f32> input_data(batch_size * dim);
  std::vector<f32> weight_data(dim);
  for (i32 b = 0; b < batch_size; ++b) {
    for (i32 d = 0; d < dim; ++d) {
      input_data[b * dim + d] = std::sin(static_cast<f32>(b * dim + d) * 0.01f);
    }
  }
  for (i32 i = 0; i < dim; ++i) {
    weight_data[i] = std::cos(static_cast<f32>(i) * 0.02f);
  }

  // Naive implementation
  RMSNormOp op_naive(dim, 1e-5f, true);
  auto weight1 = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op_naive.set_weight(std::move(weight1)));
  ASSERT_TRUE(op_naive.init());

  auto input1 = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output1 = Tensor::create({batch_size, dim}, DataType::Float32).value();
  auto result1 = op_naive.forward(input1, output1);
  ASSERT_TRUE(result1);

  // Eigen implementation
  RMSNormOp op_eigen(dim, 1e-5f, false);
  auto weight2 = Tensor::from_vector(weight_data).value();
  ASSERT_TRUE(op_eigen.set_weight(std::move(weight2)));
  ASSERT_TRUE(op_eigen.init());

  auto input2 = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output2 = Tensor::create({batch_size, dim}, DataType::Float32).value();
  auto result2 = op_eigen.forward(input2, output2);
  ASSERT_TRUE(result2);

  // Compare outputs
  auto out1 = output1.matrix_map<f32>();
  auto out2 = output2.matrix_map<f32>();

  for (i32 b = 0; b < batch_size; ++b) {
    for (i32 d = 0; d < dim; ++d) {
      EXPECT_NEAR(out1(b, d), out2(b, d), 1e-5f)
          << "Mismatch at batch " << b << ", dim " << d;
    }
  }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST(RMSNormOpTest, BenchmarkSingleVector) {
  const i32 dim = 512;
  const i32 iterations = 10000;

  // Create input and weight
  std::vector<f32> input_data(dim);
  std::vector<f32> weight_data(dim);
  for (i32 i = 0; i < dim; ++i) {
    input_data[i] = static_cast<f32>(i) * 0.01f;
    weight_data[i] = 1.0f;
  }

  // Naive benchmark
  RMSNormOp op_naive(dim, 1e-5f, true);
  auto weight_naive = Tensor::from_vector(weight_data).value();
  op_naive.set_weight(std::move(weight_naive));
  op_naive.init();

  auto input_naive = Tensor::from_vector(input_data).value();
  auto output_naive = Tensor::create({dim}, DataType::Float32).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_naive.forward(input_naive, output_naive);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Eigen benchmark
  RMSNormOp op_eigen(dim, 1e-5f, false);
  auto weight_eigen = Tensor::from_vector(weight_data).value();
  op_eigen.set_weight(std::move(weight_eigen));
  op_eigen.init();

  auto input_eigen = Tensor::from_vector(input_data).value();
  auto output_eigen = Tensor::create({dim}, DataType::Float32).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_eigen.forward(input_eigen, output_eigen);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\nRMSNorm Single Vector Benchmark (dim=" << dim << ", "
            << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << (duration_naive / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << (duration_eigen / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Speedup: " << (static_cast<double>(duration_naive) / duration_eigen)
            << "x\n";
}

TEST(RMSNormOpTest, BenchmarkBatch) {
  const i32 batch_size = 32;
  const i32 dim = 256;
  const i32 iterations = 1000;

  // Create input and weight
  std::vector<f32> input_data(batch_size * dim);
  std::vector<f32> weight_data(dim);
  for (i32 i = 0; i < batch_size * dim; ++i) {
    input_data[i] = static_cast<f32>(i) * 0.01f;
  }
  for (i32 i = 0; i < dim; ++i) {
    weight_data[i] = 1.0f;
  }

  // Naive benchmark
  RMSNormOp op_naive(dim, 1e-5f, true);
  auto weight_naive = Tensor::from_vector(weight_data).value();
  op_naive.set_weight(std::move(weight_naive));
  op_naive.init();

  auto input_naive = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output_naive = Tensor::create({batch_size, dim}, DataType::Float32).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_naive.forward(input_naive, output_naive);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Eigen benchmark
  RMSNormOp op_eigen(dim, 1e-5f, false);
  auto weight_eigen = Tensor::from_vector(weight_data).value();
  op_eigen.set_weight(std::move(weight_eigen));
  op_eigen.init();

  auto input_eigen = Tensor::from_matrix(batch_size, dim, input_data).value();
  auto output_eigen = Tensor::create({batch_size, dim}, DataType::Float32).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_eigen.forward(input_eigen, output_eigen);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\nRMSNorm Batch Benchmark (batch=" << batch_size << ", dim=" << dim
            << ", " << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << (duration_naive / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << (duration_eigen / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Speedup: " << (static_cast<double>(duration_naive) / duration_eigen)
            << "x\n";
}
