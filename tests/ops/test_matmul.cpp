/**
 * @file test_matmul.cpp
 * @brief Unit tests for MatMul operator
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include "photon/ops/matmul.hpp"

using namespace photon;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(MatMulOpTest, BasicConstruction) {
  const i32 input_dim = 512;
  const i32 output_dim = 256;

  MatMulOp op(input_dim, output_dim);

  EXPECT_EQ(op.input_dim(), input_dim);
  EXPECT_EQ(op.output_dim(), output_dim);
  EXPECT_FALSE(op.is_initialized());
  EXPECT_EQ(op.device(), DeviceType::CPU);
  EXPECT_FALSE(op.is_naive());
}

TEST(MatMulOpTest, WeightSetting) {
  const i32 input_dim = 4;
  const i32 output_dim = 3;

  MatMulOp op(input_dim, output_dim);

  // Create weight tensor [3 × 4]
  auto weight_result = Tensor::create({output_dim, input_dim}, DataType::Float32);
  ASSERT_TRUE(weight_result);

  auto weight = std::move(weight_result.value());

  // Fill with sequential values
  f32* data_ptr = weight.ptr<f32>();
  for (usize i = 0; i < weight.size(); ++i) {
    data_ptr[i] = static_cast<f32>(i);
  }

  // Set weight
  auto set_result = op.set_weight(std::move(weight));
  ASSERT_TRUE(set_result);

  // Initialize
  auto init_result = op.init();
  ASSERT_TRUE(init_result);
  EXPECT_TRUE(op.is_initialized());
}

TEST(MatMulOpTest, InvalidWeightShape) {
  const i32 input_dim = 4;
  const i32 output_dim = 3;

  MatMulOp op(input_dim, output_dim);

  // Wrong shape: 1D instead of 2D
  auto weight = Tensor::create({output_dim * input_dim}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(MatMulOpTest, InvalidWeightDimensions) {
  const i32 input_dim = 4;
  const i32 output_dim = 3;

  MatMulOp op(input_dim, output_dim);

  // Wrong dimensions: [2 × 4] instead of [3 × 4]
  auto weight = Tensor::create({2, 4}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// GEMV Tests (Vector-Matrix Multiplication)
// ============================================================================

TEST(MatMulOpTest, SimpleGEMV) {
  const i32 input_dim = 3;
  const i32 output_dim = 2;

  // Weight matrix: [2 × 3]
  // [[1, 2, 3],
  //  [4, 5, 6]]
  std::vector<f32> weight_data = {1, 2, 3, 4, 5, 6};

  // Create operator
  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input: [1, 2, 3]
  auto input = Tensor::from_vector<f32>({1.0f, 2.0f, 3.0f}).value();

  // Output: [2]
  auto output = Tensor::create({output_dim}, DataType::Float32).value();

  // Forward: [3] @ [2×3]^T -> [2]
  auto result = op.forward(input, output);
  ASSERT_TRUE(result) << result.error().to_string();

  // Expected output:
  // output[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
  // output[1] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  auto out_vec = output.vector_map<f32>();
  EXPECT_FLOAT_EQ(out_vec(0), 14.0f);
  EXPECT_FLOAT_EQ(out_vec(1), 32.0f);
}

TEST(MatMulOpTest, LargerGEMV) {
  const i32 input_dim = 128;
  const i32 output_dim = 64;

  // Create random-like weights
  std::vector<f32> weight_data(output_dim * input_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i % 100) * 0.01f;
  }

  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input vector
  std::vector<f32> input_data(input_dim);
  for (i32 i = 0; i < input_dim; ++i) {
    input_data[i] = static_cast<f32>(i) * 0.1f;
  }
  auto input = Tensor::from_vector(input_data).value();

  // Output vector
  auto output = Tensor::create({output_dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Verify output is computed (non-zero)
  auto out_vec = output.vector_map<f32>();
  bool has_nonzero = false;
  for (i32 i = 0; i < output_dim; ++i) {
    if (std::abs(out_vec(i)) > 1e-6) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// GEMM Tests (Matrix-Matrix Multiplication)
// ============================================================================

TEST(MatMulOpTest, SimpleGEMM) {
  const i32 input_dim = 3;
  const i32 output_dim = 2;
  const i32 batch_size = 2;

  // Weight matrix: [2 × 3]
  // [[1, 2, 3],
  //  [4, 5, 6]]
  std::vector<f32> weight_data = {1, 2, 3, 4, 5, 6};

  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input: [2 × 3]
  // [[1, 2, 3],
  //  [4, 5, 6]]
  std::vector<f32> input_data = {1, 2, 3, 4, 5, 6};
  auto input = Tensor::from_matrix(batch_size, input_dim, input_data).value();

  // Output: [2 × 2]
  auto output = Tensor::create({batch_size, output_dim}, DataType::Float32).value();

  // Forward: [2×3] @ [2×3]^T -> [2×2]
  auto result = op.forward(input, output);
  ASSERT_TRUE(result) << result.error().to_string();

  // Expected output:
  // Row 0: [1,2,3] @ [[1,2,3], [4,5,6]]^T = [14, 32]
  // Row 1: [4,5,6] @ [[1,2,3], [4,5,6]]^T = [32, 77]
  auto out_mat = output.matrix_map<f32>();
  EXPECT_FLOAT_EQ(out_mat(0, 0), 14.0f);
  EXPECT_FLOAT_EQ(out_mat(0, 1), 32.0f);
  EXPECT_FLOAT_EQ(out_mat(1, 0), 32.0f);
  EXPECT_FLOAT_EQ(out_mat(1, 1), 77.0f);
}

TEST(MatMulOpTest, LargerGEMM) {
  const i32 input_dim = 256;
  const i32 output_dim = 128;
  const i32 batch_size = 16;

  // Create weights
  std::vector<f32> weight_data(output_dim * input_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i % 100) * 0.01f;
  }

  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input matrix
  std::vector<f32> input_data(batch_size * input_dim);
  for (usize i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<f32>(i % 10) * 0.1f;
  }
  auto input = Tensor::from_matrix(batch_size, input_dim, input_data).value();

  // Output matrix
  auto output = Tensor::create({batch_size, output_dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Verify dimensions
  EXPECT_EQ(output.size(), static_cast<usize>(batch_size * output_dim));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(MatMulOpTest, UninitializedForward) {
  MatMulOp op(4, 3);
  // Don't set weight or init

  auto input = Tensor::from_vector<f32>({1, 2, 3, 4}).value();
  auto output = Tensor::create({3}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidOperator);
}

TEST(MatMulOpTest, WrongInputDimension) {
  const i32 input_dim = 4;
  const i32 output_dim = 3;

  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::create({output_dim, input_dim}, DataType::Float32).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Wrong input size: 5 instead of 4
  auto input = Tensor::from_vector<f32>({1, 2, 3, 4, 5}).value();
  auto output = Tensor::create({output_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

TEST(MatMulOpTest, WrongOutputDimension) {
  const i32 input_dim = 4;
  const i32 output_dim = 3;

  MatMulOp op(input_dim, output_dim);
  auto weight = Tensor::create({output_dim, input_dim}, DataType::Float32).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_vector<f32>({1, 2, 3, 4}).value();

  // Wrong output size: 5 instead of 3
  auto output = Tensor::create({5}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// Naive vs Eigen Correctness Tests
// ============================================================================

TEST(MatMulOpTest, NaiveVsEigenGEMV) {
  const i32 input_dim = 64;
  const i32 output_dim = 32;

  // Create weight
  std::vector<f32> weight_data(output_dim * input_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i) * 0.01f;
  }

  // Create input
  std::vector<f32> input_data(input_dim);
  for (i32 i = 0; i < input_dim; ++i) {
    input_data[i] = static_cast<f32>(i) * 0.1f;
  }

  // Naive implementation
  MatMulOp op_naive(input_dim, output_dim, true);  // use_naive=true
  auto weight1 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op_naive.set_weight(std::move(weight1)));
  ASSERT_TRUE(op_naive.init());

  auto input1 = Tensor::from_vector(input_data).value();
  auto output1 = Tensor::create({output_dim}, DataType::Float32).value();
  ASSERT_TRUE(op_naive.forward(input1, output1));

  // Eigen implementation
  MatMulOp op_eigen(input_dim, output_dim, false);  // use_naive=false
  auto weight2 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op_eigen.set_weight(std::move(weight2)));
  ASSERT_TRUE(op_eigen.init());

  auto input2 = Tensor::from_vector(input_data).value();
  auto output2 = Tensor::create({output_dim}, DataType::Float32).value();
  ASSERT_TRUE(op_eigen.forward(input2, output2));

  // Compare results (should be nearly identical)
  auto out1 = output1.vector_map<f32>();
  auto out2 = output2.vector_map<f32>();

  for (i32 i = 0; i < output_dim; ++i) {
    EXPECT_NEAR(out1(i), out2(i), 1e-3f) << "Mismatch at index " << i;
  }
}

TEST(MatMulOpTest, NaiveVsEigenGEMM) {
  const i32 input_dim = 32;
  const i32 output_dim = 16;
  const i32 batch_size = 4;

  // Create weight
  std::vector<f32> weight_data(output_dim * input_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i % 50) * 0.02f;
  }

  // Create input
  std::vector<f32> input_data(batch_size * input_dim);
  for (usize i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<f32>(i % 20) * 0.05f;
  }

  // Naive implementation
  MatMulOp op_naive(input_dim, output_dim, true);
  auto weight1 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op_naive.set_weight(std::move(weight1)));
  ASSERT_TRUE(op_naive.init());

  auto input1 = Tensor::from_matrix(batch_size, input_dim, input_data).value();
  auto output1 = Tensor::create({batch_size, output_dim}, DataType::Float32).value();
  ASSERT_TRUE(op_naive.forward(input1, output1));

  // Eigen implementation
  MatMulOp op_eigen(input_dim, output_dim, false);
  auto weight2 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  ASSERT_TRUE(op_eigen.set_weight(std::move(weight2)));
  ASSERT_TRUE(op_eigen.init());

  auto input2 = Tensor::from_matrix(batch_size, input_dim, input_data).value();
  auto output2 = Tensor::create({batch_size, output_dim}, DataType::Float32).value();
  ASSERT_TRUE(op_eigen.forward(input2, output2));

  // Compare results
  auto out1 = output1.matrix_map<f32>();
  auto out2 = output2.matrix_map<f32>();

  for (i32 b = 0; b < batch_size; ++b) {
    for (i32 i = 0; i < output_dim; ++i) {
      EXPECT_NEAR(out1(b, i), out2(b, i), 1e-3f)
          << "Mismatch at [" << b << ", " << i << "]";
    }
  }
}

// ============================================================================
// Concept Verification
// ============================================================================

TEST(MatMulOpTest, SatisfiesOperatorConcept) {
  static_assert(Operator<MatMulOp>, "MatMulOp must satisfy Operator concept");
  static_assert(UnaryOperator<MatMulOp>, "MatMulOp must satisfy UnaryOperator concept");

  MatMulOp op(10, 5);
  EXPECT_EQ(op.name(), "MatMulOp");
  EXPECT_EQ(op.category(), OpCategory::MatMul);
}

// ============================================================================
// Performance Benchmark Tests
// ============================================================================

TEST(MatMulOpTest, BenchmarkGEMV_Small) {
  const i32 input_dim = 512;
  const i32 output_dim = 512;
  const int iterations = 1000;

  std::vector<f32> weight_data(output_dim * input_dim, 0.01f);
  std::vector<f32> input_data(input_dim, 0.01f);

  // Benchmark Naive
  MatMulOp op_naive(input_dim, output_dim, true);
  auto w1 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  op_naive.set_weight(std::move(w1));
  op_naive.init();

  auto in1 = Tensor::from_vector(input_data).value();
  auto out1 = Tensor::create({output_dim}, DataType::Float32).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    op_naive.forward(in1, out1);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Benchmark Eigen
  MatMulOp op_eigen(input_dim, output_dim, false);
  auto w2 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  op_eigen.set_weight(std::move(w2));
  op_eigen.init();

  auto in2 = Tensor::from_vector(input_data).value();
  auto out2 = Tensor::create({output_dim}, DataType::Float32).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    op_eigen.forward(in2, out2);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\n=== GEMV Benchmark (512×512) ===" << std::endl;
  std::cout << "Naive:  " << duration_naive << " μs ("
            << duration_naive / iterations << " μs/iter)" << std::endl;
  std::cout << "Eigen:  " << duration_eigen << " μs ("
            << duration_eigen / iterations << " μs/iter)" << std::endl;
  std::cout << "Speedup: " << static_cast<double>(duration_naive) / duration_eigen
            << "x" << std::endl;

  // Eigen should be faster
  EXPECT_LT(duration_eigen, duration_naive);
}

TEST(MatMulOpTest, BenchmarkGEMM_Small) {
  const i32 input_dim = 256;
  const i32 output_dim = 256;
  const i32 batch_size = 32;
  const int iterations = 100;

  std::vector<f32> weight_data(output_dim * input_dim, 0.01f);
  std::vector<f32> input_data(batch_size * input_dim, 0.01f);

  // Benchmark Naive
  MatMulOp op_naive(input_dim, output_dim, true);
  auto w1 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  op_naive.set_weight(std::move(w1));
  op_naive.init();

  auto in1 = Tensor::from_matrix(batch_size, input_dim, input_data).value();
  auto out1 = Tensor::create({batch_size, output_dim}, DataType::Float32).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    op_naive.forward(in1, out1);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Benchmark Eigen
  MatMulOp op_eigen(input_dim, output_dim, false);
  auto w2 = Tensor::from_matrix(output_dim, input_dim, weight_data).value();
  op_eigen.set_weight(std::move(w2));
  op_eigen.init();

  auto in2 = Tensor::from_matrix(batch_size, input_dim, input_data).value();
  auto out2 = Tensor::create({batch_size, output_dim}, DataType::Float32).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    op_eigen.forward(in2, out2);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\n=== GEMM Benchmark (32×256×256) ===" << std::endl;
  std::cout << "Naive:  " << duration_naive << " μs ("
            << duration_naive / iterations << " μs/iter)" << std::endl;
  std::cout << "Eigen:  " << duration_eigen << " μs ("
            << duration_eigen / iterations << " μs/iter)" << std::endl;
  std::cout << "Speedup: " << static_cast<double>(duration_naive) / duration_eigen
            << "x" << std::endl;

  // Eigen should be significantly faster for larger matrices
  EXPECT_LT(duration_eigen, duration_naive);
}
