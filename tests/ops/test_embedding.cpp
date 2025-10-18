/**
 * @file test_embedding.cpp
 * @brief Unit tests for Embedding operator
 */

#include <gtest/gtest.h>
#include <vector>
#include "photon/ops/embedding.hpp"

using namespace photon;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(EmbeddingOpTest, BasicConstruction) {
  const i32 vocab_size = 100;
  const i32 embed_dim = 64;

  EmbeddingOp op(vocab_size, embed_dim);

  EXPECT_EQ(op.vocab_size(), vocab_size);
  EXPECT_EQ(op.embedding_dim(), embed_dim);
  EXPECT_FALSE(op.is_initialized());
  EXPECT_EQ(op.device(), DeviceType::CPU);
}

TEST(EmbeddingOpTest, WeightSetting) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  // Create weight tensor
  auto weight_result = Tensor::create({vocab_size, embed_dim}, DataType::Float32);
  ASSERT_TRUE(weight_result);

  auto weight = std::move(weight_result.value());

  // Fill with sequential values for testing
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

TEST(EmbeddingOpTest, InvalidWeightShape) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  // Wrong shape: 1D instead of 2D
  auto weight = Tensor::create({vocab_size * embed_dim}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidShape);
}

TEST(EmbeddingOpTest, InvalidWeightDimensions) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  // Wrong dimensions: [5 × 4] instead of [10 × 4]
  auto weight = Tensor::create({5, 4}, DataType::Float32).value();

  auto result = op.set_weight(std::move(weight));
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(EmbeddingOpTest, SimpleForward) {
  const i32 vocab_size = 5;
  const i32 embed_dim = 3;

  // Create operator
  EmbeddingOp op(vocab_size, embed_dim);

  // Create weight: each row is [row_idx*3, row_idx*3+1, row_idx*3+2]
  std::vector<f32> weight_data;
  for (i32 i = 0; i < vocab_size; ++i) {
    for (i32 j = 0; j < embed_dim; ++j) {
      weight_data.push_back(static_cast<f32>(i * embed_dim + j));
    }
  }

  auto weight = Tensor::from_matrix(vocab_size, embed_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Input tokens: [0, 2, 4]
  std::vector<i32> tokens = {0, 2, 4};
  auto input = Tensor::from_vector(tokens).value();

  // Output tensor
  auto output = Tensor::create({3, embed_dim}, DataType::Float32).value();

  // Forward
  auto result = op.forward(input, output);
  ASSERT_TRUE(result) << result.error().to_string();

  // Verify output
  auto out_mat = output.matrix_map<f32>();

  // Token 0: [0, 1, 2]
  EXPECT_FLOAT_EQ(out_mat(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(out_mat(0, 1), 1.0f);
  EXPECT_FLOAT_EQ(out_mat(0, 2), 2.0f);

  // Token 2: [6, 7, 8]
  EXPECT_FLOAT_EQ(out_mat(1, 0), 6.0f);
  EXPECT_FLOAT_EQ(out_mat(1, 1), 7.0f);
  EXPECT_FLOAT_EQ(out_mat(1, 2), 8.0f);

  // Token 4: [12, 13, 14]
  EXPECT_FLOAT_EQ(out_mat(2, 0), 12.0f);
  EXPECT_FLOAT_EQ(out_mat(2, 1), 13.0f);
  EXPECT_FLOAT_EQ(out_mat(2, 2), 14.0f);
}

TEST(EmbeddingOpTest, SingleToken) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  // Simple weight
  std::vector<f32> weight_data(vocab_size * embed_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i);
  }

  auto weight = Tensor::from_matrix(vocab_size, embed_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Single token
  auto input = Tensor::from_vector<i32>({5}).value();
  auto output = Tensor::create({1, embed_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Token 5 should give [20, 21, 22, 23]
  auto out_vec = output.vector_map<f32>();
  EXPECT_FLOAT_EQ(out_vec(0), 20.0f);
  EXPECT_FLOAT_EQ(out_vec(1), 21.0f);
  EXPECT_FLOAT_EQ(out_vec(2), 22.0f);
  EXPECT_FLOAT_EQ(out_vec(3), 23.0f);
}

TEST(EmbeddingOpTest, LargerBatch) {
  const i32 vocab_size = 100;
  const i32 embed_dim = 16;
  const i32 batch_size = 32;

  EmbeddingOp op(vocab_size, embed_dim);

  // Random-like weight
  std::vector<f32> weight_data(vocab_size * embed_dim);
  for (usize i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = static_cast<f32>(i % 100) * 0.1f;
  }

  auto weight = Tensor::from_matrix(vocab_size, embed_dim, weight_data).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Batch of tokens
  std::vector<i32> tokens(batch_size);
  for (i32 i = 0; i < batch_size; ++i) {
    tokens[i] = i % vocab_size;
  }

  auto input = Tensor::from_vector(tokens).value();
  auto output = Tensor::create({batch_size, embed_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  ASSERT_TRUE(result);

  // Basic sanity check
  EXPECT_EQ(output.size(), static_cast<usize>(batch_size * embed_dim));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(EmbeddingOpTest, TokenOutOfBounds) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  auto weight = Tensor::create({vocab_size, embed_dim}, DataType::Float32).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Token 15 is out of bounds (vocab_size=10)
  auto input = Tensor::from_vector<i32>({5, 15, 3}).value();
  auto output = Tensor::create({3, embed_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidIndex);
}

TEST(EmbeddingOpTest, NegativeToken) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  auto weight = Tensor::create({vocab_size, embed_dim}, DataType::Float32).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  // Negative token
  auto input = Tensor::from_vector<i32>({-1}).value();
  auto output = Tensor::create({1, embed_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidIndex);
}

TEST(EmbeddingOpTest, UninitializedForward) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);
  // Don't set weight or init

  auto input = Tensor::from_vector<i32>({0, 1, 2}).value();
  auto output = Tensor::create({3, embed_dim}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidOperator);
}

TEST(EmbeddingOpTest, WrongOutputShape) {
  const i32 vocab_size = 10;
  const i32 embed_dim = 4;

  EmbeddingOp op(vocab_size, embed_dim);

  auto weight = Tensor::create({vocab_size, embed_dim}, DataType::Float32).value();
  ASSERT_TRUE(op.set_weight(std::move(weight)));
  ASSERT_TRUE(op.init());

  auto input = Tensor::from_vector<i32>({0, 1, 2}).value();

  // Wrong output shape: [3 × 8] instead of [3 × 4]
  auto output = Tensor::create({3, 8}, DataType::Float32).value();

  auto result = op.forward(input, output);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::ShapeMismatch);
}

// ============================================================================
// Concept Verification Tests
// ============================================================================

TEST(EmbeddingOpTest, SatisfiesOperatorConcept) {
  // This test will fail to compile if EmbeddingOp doesn't satisfy the Operator concept
  static_assert(Operator<EmbeddingOp>, "EmbeddingOp must satisfy Operator concept");
  static_assert(UnaryOperator<EmbeddingOp>, "EmbeddingOp must satisfy UnaryOperator concept");

  EmbeddingOp op(10, 4);
  EXPECT_EQ(op.name(), "EmbeddingOp");
  EXPECT_EQ(op.category(), OpCategory::Embedding);
}
