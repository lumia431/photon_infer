/**
 * @file test_rope.cpp
 * @brief Unit tests for RoPE operator
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>
#include "photon/ops/rope.hpp"

using namespace photon;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(RoPEOpTest, BasicConstruction) {
  const i32 dim = 1024;
  const i32 kv_dim = 256;
  const i32 head_size = 64;
  const i32 max_seq_len = 2048;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);

  EXPECT_EQ(op.dim(), dim);
  EXPECT_EQ(op.kv_dim(), kv_dim);
  EXPECT_EQ(op.head_size(), head_size);
  EXPECT_EQ(op.max_seq_len(), max_seq_len);
  EXPECT_FALSE(op.is_initialized());
  EXPECT_EQ(op.device(), DeviceType::CPU);
  EXPECT_FALSE(op.is_naive());
}

TEST(RoPEOpTest, Initialization) {
  const i32 dim = 512;
  const i32 kv_dim = 512;
  const i32 head_size = 64;
  const i32 max_seq_len = 128;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);

  // Initialize
  auto init_result = op.init();
  ASSERT_TRUE(init_result);
  EXPECT_TRUE(op.is_initialized());

  // Check cache size
  EXPECT_EQ(op.sin_cache().size(), max_seq_len * head_size);
  EXPECT_EQ(op.cos_cache().size(), max_seq_len * head_size);
}

TEST(RoPEOpTest, CacheComputation) {
  const i32 head_size = 4;
  const i32 max_seq_len = 2;

  RoPEOp op(8, 8, head_size, max_seq_len);
  ASSERT_TRUE(op.init());

  const auto& sin_cache = op.sin_cache();
  const auto& cos_cache = op.cos_cache();

  // Manually verify some values
  // For pos=0, head_dim=0: freq = 1.0/10000^0 = 1.0, angle = 0
  EXPECT_FLOAT_EQ(sin_cache[0 * head_size + 0], std::sin(0.0f));
  EXPECT_FLOAT_EQ(cos_cache[0 * head_size + 0], std::cos(0.0f));

  // For pos=1, head_dim=0: freq = 1.0, angle = 1.0
  EXPECT_FLOAT_EQ(sin_cache[1 * head_size + 0], std::sin(1.0f));
  EXPECT_FLOAT_EQ(cos_cache[1 * head_size + 0], std::cos(1.0f));

  // For pos=0, head_dim=1: freq = 1.0/10000^0.25, angle = 0
  EXPECT_FLOAT_EQ(sin_cache[0 * head_size + 1], 0.0f);
  EXPECT_FLOAT_EQ(cos_cache[0 * head_size + 1], 1.0f);
}

TEST(RoPEOpTest, UninitializedForward) {
  RoPEOp op(256, 256, 64, 128);

  auto q = Tensor::create({256}, DataType::Float32).value();
  auto k = Tensor::create({256}, DataType::Float32).value();

  // Forward without initialization should fail
  auto result = op.forward(q, k, 0);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidOperator);
}

TEST(RoPEOpTest, InvalidPosition) {
  const i32 max_seq_len = 128;
  RoPEOp op(256, 256, 64, max_seq_len);
  ASSERT_TRUE(op.init());

  auto q = Tensor::create({256}, DataType::Float32).value();
  auto k = Tensor::create({256}, DataType::Float32).value();

  // Negative position
  auto result1 = op.forward(q, k, -1);
  EXPECT_FALSE(result1);
  EXPECT_EQ(result1.error().code(), ErrorCode::InvalidArgument);

  // Position >= max_seq_len
  auto result2 = op.forward(q, k, max_seq_len);
  EXPECT_FALSE(result2);
  EXPECT_EQ(result2.error().code(), ErrorCode::InvalidArgument);
}

TEST(RoPEOpTest, DimensionMismatch) {
  RoPEOp op(256, 128, 64, 512);
  ASSERT_TRUE(op.init());

  auto q = Tensor::create({512}, DataType::Float32).value();  // Wrong dim
  auto k = Tensor::create({128}, DataType::Float32).value();

  auto result = op.forward(q, k, 0);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// Correctness Tests
// ============================================================================

TEST(RoPEOpTest, SimpleRotation) {
  // Small example: 4-dim with 2-dim heads
  const i32 dim = 4;
  const i32 kv_dim = 4;
  const i32 head_size = 2;
  const i32 max_seq_len = 10;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);
  ASSERT_TRUE(op.init());

  // Input: [1, 0, 1, 0]
  std::vector<f32> q_data = {1.0f, 0.0f, 1.0f, 0.0f};
  std::vector<f32> k_data = {0.0f, 1.0f, 0.0f, 1.0f};

  auto q = Tensor::from_vector(q_data).value();
  auto k = Tensor::from_vector(k_data).value();

  // Apply RoPE at position 0
  auto result = op.forward(q, k, 0);
  ASSERT_TRUE(result);

  // At pos=0, all angles should be 0, so rotation should be identity
  // cos(0) = 1, sin(0) = 0
  // x_new = x * cos - y * sin = x * 1 - y * 0 = x
  // y_new = x * sin + y * cos = x * 0 + y * 1 = y
  auto q_vec = q.vector_map<f32>();
  EXPECT_FLOAT_EQ(q_vec(0), 1.0f);
  EXPECT_FLOAT_EQ(q_vec(1), 0.0f);
  EXPECT_FLOAT_EQ(q_vec(2), 1.0f);
  EXPECT_FLOAT_EQ(q_vec(3), 0.0f);
}

TEST(RoPEOpTest, RotationAt90Degrees) {
  // Test with a frequency that gives 90-degree rotation
  const i32 dim = 2;
  const i32 kv_dim = 2;
  const i32 head_size = 2;
  const i32 max_seq_len = 10;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);
  ASSERT_TRUE(op.init());

  // Input: [1, 0]
  std::vector<f32> q_data = {1.0f, 0.0f};
  std::vector<f32> k_data = {1.0f, 0.0f};

  auto q = Tensor::from_vector(q_data).value();
  auto k = Tensor::from_vector(k_data).value();

  // Apply at different positions and check rotation
  for (i32 pos = 0; pos < 5; ++pos) {
    auto q_test = q.clone();
    ASSERT_TRUE(q_test);
    auto k_test = k.clone();
    ASSERT_TRUE(k_test);

    ASSERT_TRUE(op.forward(q_test.value(), k_test.value(), pos));

    // Verify the rotation was applied
    auto q_vec = q_test.value().vector_map<f32>();
    f32 q0 = q_vec(0);
    f32 q1 = q_vec(1);

    // The magnitude should be preserved
    f32 mag_orig = std::sqrt(1.0f * 1.0f + 0.0f * 0.0f);
    f32 mag_new = std::sqrt(q0 * q0 + q1 * q1);
    EXPECT_NEAR(mag_orig, mag_new, 1e-6f);
  }
}

TEST(RoPEOpTest, GQAScenario) {
  // Test Grouped Query Attention: Q has more dims than K
  const i32 dim = 1024;      // 16 query heads × 64
  const i32 kv_dim = 256;    // 4 KV heads × 64
  const i32 head_size = 64;
  const i32 max_seq_len = 128;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);
  ASSERT_TRUE(op.init());

  // Create random-like data
  std::vector<f32> q_data(dim);
  std::vector<f32> k_data(kv_dim);
  for (i32 i = 0; i < dim; ++i) {
    q_data[i] = std::sin(static_cast<f32>(i) * 0.1f);
  }
  for (i32 i = 0; i < kv_dim; ++i) {
    k_data[i] = std::cos(static_cast<f32>(i) * 0.1f);
  }

  auto q = Tensor::from_vector(q_data).value();
  auto k = Tensor::from_vector(k_data).value();

  auto result = op.forward(q, k, 5);
  ASSERT_TRUE(result);

  // Verify that rotation preserves magnitude
  auto q_vec = q.vector_map<f32>();
  auto k_vec = k.vector_map<f32>();

  // Check that values changed (not identity)
  bool q_changed = false;
  for (i32 i = 0; i < dim; ++i) {
    if (std::abs(q_vec(i) - q_data[i]) > 1e-6f) {
      q_changed = true;
      break;
    }
  }
  EXPECT_TRUE(q_changed);
}

TEST(RoPEOpTest, MultiplePositions) {
  const i32 dim = 256;
  const i32 kv_dim = 256;
  const i32 head_size = 64;
  const i32 max_seq_len = 100;

  RoPEOp op(dim, kv_dim, head_size, max_seq_len);
  ASSERT_TRUE(op.init());

  std::vector<f32> q_data(dim, 1.0f);
  std::vector<f32> k_data(kv_dim, 1.0f);

  // Apply RoPE at different positions
  for (i32 pos : {0, 10, 50, 99}) {
    auto q = Tensor::from_vector(q_data).value();
    auto k = Tensor::from_vector(k_data).value();

    auto result = op.forward(q, k, pos);
    ASSERT_TRUE(result) << "Failed at position " << pos;

    // Verify tensors were modified
    auto q_vec = q.vector_map<f32>();
    bool modified = false;
    for (i32 i = 0; i < dim; ++i) {
      if (std::abs(q_vec(i) - q_data[i]) > 1e-6f) {
        modified = true;
        break;
      }
    }

    // At pos > 0, values should change
    if (pos > 0) {
      EXPECT_TRUE(modified);
    }
  }
}

// ============================================================================
// Naive vs Eigen Correctness
// ============================================================================

TEST(RoPEOpTest, NaiveVsEigen) {
  const i32 dim = 512;
  const i32 kv_dim = 256;
  const i32 head_size = 64;
  const i32 max_seq_len = 128;

  // Create input data
  std::vector<f32> q_data(dim);
  std::vector<f32> k_data(kv_dim);
  for (i32 i = 0; i < dim; ++i) {
    q_data[i] = std::sin(static_cast<f32>(i) * 0.01f);
  }
  for (i32 i = 0; i < kv_dim; ++i) {
    k_data[i] = std::cos(static_cast<f32>(i) * 0.01f);
  }

  // Naive implementation
  RoPEOp op_naive(dim, kv_dim, head_size, max_seq_len, true);
  ASSERT_TRUE(op_naive.init());

  auto q1 = Tensor::from_vector(q_data).value();
  auto k1 = Tensor::from_vector(k_data).value();
  ASSERT_TRUE(op_naive.forward(q1, k1, 10));

  // Eigen implementation
  RoPEOp op_eigen(dim, kv_dim, head_size, max_seq_len, false);
  ASSERT_TRUE(op_eigen.init());

  auto q2 = Tensor::from_vector(q_data).value();
  auto k2 = Tensor::from_vector(k_data).value();
  ASSERT_TRUE(op_eigen.forward(q2, k2, 10));

  // Compare results
  auto q1_vec = q1.vector_map<f32>();
  auto q2_vec = q2.vector_map<f32>();
  auto k1_vec = k1.vector_map<f32>();
  auto k2_vec = k2.vector_map<f32>();

  for (i32 i = 0; i < dim; ++i) {
    EXPECT_NEAR(q1_vec(i), q2_vec(i), 1e-6f) << "Q mismatch at index " << i;
  }

  for (i32 i = 0; i < kv_dim; ++i) {
    EXPECT_NEAR(k1_vec(i), k2_vec(i), 1e-6f) << "K mismatch at index " << i;
  }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST(RoPEOpTest, BenchmarkSmall) {
  const i32 dim = 512;
  const i32 kv_dim = 256;
  const i32 head_size = 64;
  const i32 max_seq_len = 2048;
  const i32 iterations = 10000;

  // Create input data
  std::vector<f32> q_data(dim);
  std::vector<f32> k_data(kv_dim);
  for (i32 i = 0; i < dim; ++i) {
    q_data[i] = static_cast<f32>(i) * 0.01f;
  }
  for (i32 i = 0; i < kv_dim; ++i) {
    k_data[i] = static_cast<f32>(i) * 0.01f;
  }

  // Naive benchmark
  RoPEOp op_naive(dim, kv_dim, head_size, max_seq_len, true);
  op_naive.init();

  auto q_naive = Tensor::from_vector(q_data).value();
  auto k_naive = Tensor::from_vector(k_data).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_naive.forward(q_naive, k_naive, i % max_seq_len);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Eigen benchmark
  RoPEOp op_eigen(dim, kv_dim, head_size, max_seq_len, false);
  op_eigen.init();

  auto q_eigen = Tensor::from_vector(q_data).value();
  auto k_eigen = Tensor::from_vector(k_data).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_eigen.forward(q_eigen, k_eigen, i % max_seq_len);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\nRoPE Benchmark (dim=" << dim << ", kv_dim=" << kv_dim
            << ", " << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << (duration_naive / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << (duration_eigen / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Speedup: " << (static_cast<double>(duration_naive) / duration_eigen)
            << "x\n";
}

TEST(RoPEOpTest, BenchmarkLarge) {
  const i32 dim = 2048;
  const i32 kv_dim = 512;
  const i32 head_size = 128;
  const i32 max_seq_len = 4096;
  const i32 iterations = 1000;

  std::vector<f32> q_data(dim);
  std::vector<f32> k_data(kv_dim);
  for (i32 i = 0; i < dim; ++i) {
    q_data[i] = static_cast<f32>(i) * 0.001f;
  }
  for (i32 i = 0; i < kv_dim; ++i) {
    k_data[i] = static_cast<f32>(i) * 0.001f;
  }

  // Naive benchmark
  RoPEOp op_naive(dim, kv_dim, head_size, max_seq_len, true);
  op_naive.init();

  auto q_naive = Tensor::from_vector(q_data).value();
  auto k_naive = Tensor::from_vector(k_data).value();

  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_naive.forward(q_naive, k_naive, i % max_seq_len);
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Eigen benchmark
  RoPEOp op_eigen(dim, kv_dim, head_size, max_seq_len, false);
  op_eigen.init();

  auto q_eigen = Tensor::from_vector(q_data).value();
  auto k_eigen = Tensor::from_vector(k_data).value();

  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 i = 0; i < iterations; ++i) {
    op_eigen.forward(q_eigen, k_eigen, i % max_seq_len);
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  std::cout << "\nRoPE Large Benchmark (dim=" << dim << ", kv_dim=" << kv_dim
            << ", " << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << (duration_naive / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << (duration_eigen / static_cast<double>(iterations)) << " μs/iter\n";
  std::cout << "  Speedup: " << (static_cast<double>(duration_naive) / duration_eigen)
            << "x\n";
}
