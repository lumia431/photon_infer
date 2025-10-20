#include <gtest/gtest.h>
#include <photon/ops/mha.hpp>
#include <photon/core/tensor.hpp>
#include <cmath>
#include <random>

using namespace photon;

// Test 1: Basic construction
TEST(MHAOpTest, BasicConstruction) {
  const i32 head_num = 2;
  const i32 head_size = 4;
  const i32 dim = head_num * head_size;  // 8
  const i32 kv_dim = dim;
  const i32 seq_len = 4;

  MHAOp op(dim, kv_dim, head_num, head_size, seq_len);

  EXPECT_EQ(op.dim(), dim);
  EXPECT_EQ(op.kv_dim(), kv_dim);
  EXPECT_EQ(op.head_num(), head_num);
  EXPECT_EQ(op.head_size(), head_size);
  EXPECT_EQ(op.seq_len(), seq_len);
  EXPECT_EQ(op.kv_mul(), 1);  // No GQA
  EXPECT_FALSE(op.use_naive());
}

// Test 2: Simple single-head attention at position 0
TEST(MHAOpTest, SingleHeadPosition0) {
  const i32 head_num = 1;
  const i32 head_size = 4;
  const i32 dim = head_num * head_size;  // 4
  const i32 kv_dim = dim;
  const i32 seq_len = 4;
  const i32 pos = 0;

  MHAOp op(dim, kv_dim, head_num, head_size, seq_len, /*use_naive=*/true);
  ASSERT_TRUE(op.init());

  auto query = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(query);
  auto key_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(key_cache);
  auto value_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(value_cache);
  auto output = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // Initialize data: Q=[1,0,0,0], K[0]=[1,0,0,0], V[0]=[2,3,4,5]
  f32* q_ptr = query.value().ptr<f32>();
  q_ptr[0] = 1.0f;
  q_ptr[1] = 0.0f;
  q_ptr[2] = 0.0f;
  q_ptr[3] = 0.0f;

  f32* k_ptr = key_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    k_ptr[i] = 0.0f;
  }
  k_ptr[0] = 1.0f;  // K[0][0] = 1

  f32* v_ptr = value_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    v_ptr[i] = 0.0f;
  }
  v_ptr[0] = 2.0f;
  v_ptr[1] = 3.0f;
  v_ptr[2] = 4.0f;
  v_ptr[3] = 5.0f;

  // Forward pass
  ASSERT_TRUE(op.forward(query.value(), key_cache.value(),
                        value_cache.value(), output.value(), pos));

  // At pos=0, attention only over first position
  // Output = 1.0 * V[0] = [2,3,4,5]
  f32* out_ptr = output.value().ptr<f32>();
  EXPECT_NEAR(out_ptr[0], 2.0f, 1e-4f);
  EXPECT_NEAR(out_ptr[1], 3.0f, 1e-4f);
  EXPECT_NEAR(out_ptr[2], 4.0f, 1e-4f);
  EXPECT_NEAR(out_ptr[3], 5.0f, 1e-4f);
}

// Test 3: Two-head attention
TEST(MHAOpTest, TwoHeads) {
  const i32 head_num = 2;
  const i32 head_size = 3;
  const i32 dim = head_num * head_size;  // 6
  const i32 kv_dim = dim;
  const i32 seq_len = 4;
  const i32 pos = 1;

  MHAOp op(dim, kv_dim, head_num, head_size, seq_len, /*use_naive=*/true);
  ASSERT_TRUE(op.init());

  auto query = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(query);
  auto key_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(key_cache);
  auto value_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(value_cache);
  auto output = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // Initialize with simple values
  f32* q_ptr = query.value().ptr<f32>();
  for (i32 i = 0; i < dim; ++i) {
    q_ptr[i] = 1.0f;
  }

  f32* k_ptr = key_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    k_ptr[i] = 1.0f;
  }

  f32* v_ptr = value_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    v_ptr[i] = static_cast<f32>(i);
  }

  ASSERT_TRUE(op.forward(query.value(), key_cache.value(),
                        value_cache.value(), output.value(), pos));

  // Output should not be all zeros
  f32* out_ptr = output.value().ptr<f32>();
  f32 sum = 0.0f;
  for (i32 i = 0; i < dim; ++i) {
    sum += std::abs(out_ptr[i]);
  }
  EXPECT_GT(sum, 0.1f);
}

// Test 4: Naive vs Eigen implementation
TEST(MHAOpTest, NaiveVsEigen) {
  const i32 head_num = 2;
  const i32 head_size = 8;
  const i32 dim = 16;
  const i32 kv_dim = 16;
  const i32 seq_len = 16;
  const i32 pos = 7;

  MHAOp op_naive(dim, kv_dim, head_num, head_size, seq_len, /*use_naive=*/true);
  MHAOp op_eigen(dim, kv_dim, head_num, head_size, seq_len, /*use_naive=*/false);
  ASSERT_TRUE(op_naive.init());
  ASSERT_TRUE(op_eigen.init());

  auto query = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(query);
  auto key_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(key_cache);
  auto value_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(value_cache);
  auto output_naive = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_naive);
  auto output_eigen = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_eigen);

  // Random initialization
  std::mt19937 gen(123);
  std::normal_distribution<f32> dis(0.0f, 1.0f);

  f32* q_ptr = query.value().ptr<f32>();
  for (i32 i = 0; i < dim; ++i) {
    q_ptr[i] = dis(gen);
  }

  f32* k_ptr = key_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    k_ptr[i] = dis(gen);
  }

  f32* v_ptr = value_cache.value().ptr<f32>();
  for (i32 i = 0; i < seq_len * kv_dim; ++i) {
    v_ptr[i] = dis(gen);
  }

  ASSERT_TRUE(op_naive.forward(query.value(), key_cache.value(),
                              value_cache.value(), output_naive.value(), pos));
  ASSERT_TRUE(op_eigen.forward(query.value(), key_cache.value(),
                              value_cache.value(), output_eigen.value(), pos));

  // Compare outputs
  f32* out_naive_ptr = output_naive.value().ptr<f32>();
  f32* out_eigen_ptr = output_eigen.value().ptr<f32>();

  for (i32 i = 0; i < dim; ++i) {
    EXPECT_NEAR(out_naive_ptr[i], out_eigen_ptr[i], 1e-3f)
        << "Mismatch at index " << i;
  }
}

// Test 5: Error handling - empty tensors
TEST(MHAOpTest, ErrorHandlingEmptyTensors) {
  const i32 head_num = 1;
  const i32 head_size = 4;
  const i32 dim = 4;
  const i32 kv_dim = 4;
  const i32 seq_len = 4;

  MHAOp op(dim, kv_dim, head_num, head_size, seq_len);
  ASSERT_TRUE(op.init());

  auto query = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(query);
  auto key_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(key_cache);
  auto value_cache = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(value_cache);
  auto output = Tensor::create({dim}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  Tensor empty_tensor;

  // Empty query
  EXPECT_FALSE(op.forward(empty_tensor, key_cache.value(),
                         value_cache.value(), output.value(), 0));

  // Empty key cache
  EXPECT_FALSE(op.forward(query.value(), empty_tensor,
                         value_cache.value(), output.value(), 0));

  // Empty value cache
  EXPECT_FALSE(op.forward(query.value(), key_cache.value(),
                         empty_tensor, output.value(), 0));

  // Empty output
  EXPECT_FALSE(op.forward(query.value(), key_cache.value(),
                         value_cache.value(), empty_tensor, 0));
}
