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

// Test 3: Two-head attention with verifiable computation
TEST(MHAOpTest, TwoHeads) {
  const i32 head_num = 2;
  const i32 head_size = 2;
  const i32 dim = head_num * head_size;  // 4
  const i32 kv_dim = dim;
  const i32 seq_len = 3;
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

  // Initialize with carefully chosen values for verifiable computation
  // Q = [1, 0, 0, 1]  // head0: [1,0], head1: [0,1]
  f32* q_ptr = query.value().ptr<f32>();
  q_ptr[0] = 1.0f;  // head0[0]
  q_ptr[1] = 0.0f;  // head0[1]
  q_ptr[2] = 0.0f;  // head1[0]
  q_ptr[3] = 1.0f;  // head1[1]

  // Key cache: positions 0,1,2 with different values per head
  // pos0: [1,0, 0,1]  // head0: [1,0], head1: [0,1]
  // pos1: [0,1, 1,0]  // head0: [0,1], head1: [1,0]
  // pos2: [1,1, 1,1]  // unused
  f32* k_ptr = key_cache.value().ptr<f32>();
  // pos 0
  k_ptr[0] = 1.0f; k_ptr[1] = 0.0f; k_ptr[2] = 0.0f; k_ptr[3] = 1.0f;
  // pos 1
  k_ptr[4] = 0.0f; k_ptr[5] = 1.0f; k_ptr[6] = 1.0f; k_ptr[7] = 0.0f;
  // pos 2
  k_ptr[8] = 1.0f; k_ptr[9] = 1.0f; k_ptr[10] = 1.0f; k_ptr[11] = 1.0f;

  // Value cache: different values per position and head
  // pos0: [2,3, 4,5]  // head0: [2,3], head1: [4,5]
  // pos1: [6,7, 8,9]  // head0: [6,7], head1: [8,9]
  // pos2: [0,0, 0,0]  // unused
  f32* v_ptr = value_cache.value().ptr<f32>();
  // pos 0
  v_ptr[0] = 2.0f; v_ptr[1] = 3.0f; v_ptr[2] = 4.0f; v_ptr[3] = 5.0f;
  // pos 1
  v_ptr[4] = 6.0f; v_ptr[5] = 7.0f; v_ptr[6] = 8.0f; v_ptr[7] = 9.0f;
  // pos 2
  v_ptr[8] = 0.0f; v_ptr[9] = 0.0f; v_ptr[10] = 0.0f; v_ptr[11] = 0.0f;

  ASSERT_TRUE(op.forward(query.value(), key_cache.value(),
                        value_cache.value(), output.value(), pos));

  // Verify output with manually computed expected values
  f32* out_ptr = output.value().ptr<f32>();

  // Head 0 computation:
  // score[0] = (Q[0]·K[0]) / sqrt(2) = ([1,0]·[1,0]) / sqrt(2) = 1/sqrt(2) ≈ 0.707
  // score[1] = (Q[0]·K[1]) / sqrt(2) = ([1,0]·[0,1]) / sqrt(2) = 0
  // softmax([0.707, 0]) ≈ [0.670, 0.330]
  // output[0] = 0.670*[2,3] + 0.330*[6,7] ≈ [3.34, 4.34]
  const f32 sqrt2 = std::sqrt(2.0f);
  const f32 score0 = 1.0f / sqrt2;
  const f32 score1 = 0.0f;
  const f32 max_score = std::max(score0, score1);
  const f32 exp0 = std::exp(score0 - max_score);
  const f32 exp1 = std::exp(score1 - max_score);
  const f32 sum_exp = exp0 + exp1;
  const f32 attn0 = exp0 / sum_exp;
  const f32 attn1 = exp1 / sum_exp;

  const f32 expected_head0_0 = attn0 * 2.0f + attn1 * 6.0f;
  const f32 expected_head0_1 = attn0 * 3.0f + attn1 * 7.0f;

  EXPECT_NEAR(out_ptr[0], expected_head0_0, 1e-3f);
  EXPECT_NEAR(out_ptr[1], expected_head0_1, 1e-3f);

  // Head 1 computation:
  // score[0] = (Q[1]·K[0]) / sqrt(2) = ([0,1]·[0,1]) / sqrt(2) = 1/sqrt(2) ≈ 0.707
  // score[1] = (Q[1]·K[1]) / sqrt(2) = ([0,1]·[1,0]) / sqrt(2) = 0
  // softmax same as head 0: [0.670, 0.330]
  // output[1] = 0.670*[4,5] + 0.330*[8,9] ≈ [5.34, 6.34]
  const f32 expected_head1_0 = attn0 * 4.0f + attn1 * 8.0f;
  const f32 expected_head1_1 = attn0 * 5.0f + attn1 * 9.0f;

  EXPECT_NEAR(out_ptr[2], expected_head1_0, 1e-3f);
  EXPECT_NEAR(out_ptr[3], expected_head1_1, 1e-3f);
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
