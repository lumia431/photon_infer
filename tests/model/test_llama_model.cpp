/**
 * @file test_llama_model.cpp
 * @brief Unit tests for LLaMA model
 */

#include <gtest/gtest.h>
#include <photon/arch/llama_model.hpp>
#include <photon/arch/transformer_block.hpp>
#include <random>

using namespace photon;
using namespace photon::model;

// Helper function to create a small test config
TransformerConfig create_test_config() {
  TransformerConfig config;
  config.vocab_size = 1000;
  config.dim = 128;
  config.hidden_dim = 256;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;  // GQA
  config.head_size = 32;
  config.seq_len = 64;
  config.norm_eps = 1e-5f;
  config.compute_derived();
  return config;
}

// Helper to initialize random weights
Tensor create_random_weight(std::vector<i32> shape, std::mt19937& gen) {
  auto tensor = Tensor::create(shape, DataType::Float32, DeviceType::CPU);
  EXPECT_TRUE(tensor);

  std::normal_distribution<f32> dis(0.0f, 0.02f);
  f32* ptr = tensor.value().ptr<f32>();
  for (usize i = 0; i < tensor.value().size(); ++i) {
    ptr[i] = dis(gen);
  }

  return std::move(tensor.value());
}

// Test 1: Model construction
TEST(LLaMAModelTest, Construction) {
  auto config = create_test_config();
  LLaMAModel model(config);

  EXPECT_EQ(model.config().vocab_size, 1000);
  EXPECT_EQ(model.config().dim, 128);
  EXPECT_EQ(model.config().n_layers, 2);
}

// Test 2: Model initialization
TEST(LLaMAModelTest, Initialization) {
  auto config = create_test_config();
  LLaMAModel model(config);

  std::mt19937 gen(42);

  // Set embedding weight before init (required)
  auto emb_weight = create_random_weight({config.vocab_size, config.dim}, gen);
  ASSERT_TRUE(model.set_embedding(std::move(emb_weight)));

  auto init_result = model.init();
  ASSERT_TRUE(init_result) << init_result.error().message();
}

// Test 3: Weight setting
TEST(LLaMAModelTest, WeightSetting) {
  auto config = create_test_config();
  LLaMAModel model(config);
  ASSERT_TRUE(model.init());

  std::mt19937 gen(42);

  // Set embedding
  auto emb_weight = create_random_weight({config.vocab_size, config.dim}, gen);
  EXPECT_TRUE(model.set_embedding(std::move(emb_weight)));

  // Set final norm
  auto norm_weight = create_random_weight({config.dim}, gen);
  EXPECT_TRUE(model.set_final_norm(std::move(norm_weight)));

  // Set classifier
  auto cls_weight = create_random_weight({config.vocab_size, config.dim}, gen);
  auto cls_result = model.set_classifier(std::move(cls_weight));
  EXPECT_TRUE(cls_result) << cls_result.error().message();

  // Set weights for first block
  auto& block0 = model.get_block(0);

  EXPECT_TRUE(block0.set_wq(create_random_weight({config.dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_wk(create_random_weight({config.kv_dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_wv(create_random_weight({config.kv_dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_wo(create_random_weight({config.dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_w1(create_random_weight({config.hidden_dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_w2(create_random_weight({config.dim, config.hidden_dim}, gen)));
  EXPECT_TRUE(block0.set_w3(create_random_weight({config.hidden_dim, config.dim}, gen)));
  EXPECT_TRUE(block0.set_attn_norm(create_random_weight({config.dim}, gen)));
  EXPECT_TRUE(block0.set_ffn_norm(create_random_weight({config.dim}, gen)));
}

// Test 4: Forward pass (single token)
TEST(LLaMAModelTest, ForwardPass) {
  auto config = create_test_config();
  LLaMAModel model(config);

  std::mt19937 gen(123);

  // Initialize all weights
  ASSERT_TRUE(model.set_embedding(create_random_weight({config.vocab_size, config.dim}, gen)));
  ASSERT_TRUE(model.set_final_norm(create_random_weight({config.dim}, gen)));
  ASSERT_TRUE(model.set_classifier(create_random_weight({config.vocab_size, config.dim}, gen)));

  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto& block = model.get_block(layer);
    ASSERT_TRUE(block.set_wq(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wk(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wv(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wo(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w1(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w2(create_random_weight({config.dim, config.hidden_dim}, gen)));
    ASSERT_TRUE(block.set_w3(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_attn_norm(create_random_weight({config.dim}, gen)));
    ASSERT_TRUE(block.set_ffn_norm(create_random_weight({config.dim}, gen)));
  }

  // Initialize model (allocates buffers after weights are set)
  ASSERT_TRUE(model.init());

  // Create logits buffer
  auto logits = Tensor::create({config.vocab_size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(logits);

  // Forward pass with token 5 at position 0
  auto fwd_result = model.forward(5, 0, logits.value());
  ASSERT_TRUE(fwd_result) << fwd_result.error().message();

  // Verify logits are computed (non-zero)
  f32* logits_ptr = logits.value().ptr<f32>();
  f32 sum = 0.0f;
  for (i32 i = 0; i < config.vocab_size; ++i) {
    sum += std::abs(logits_ptr[i]);
  }
  EXPECT_GT(sum, 0.01f);
}

// Test 5: Generate next token
TEST(LLaMAModelTest, GenerateNext) {
  auto config = create_test_config();
  LLaMAModel model(config);

  std::mt19937 gen(456);

  // Initialize all weights
  ASSERT_TRUE(model.set_embedding(create_random_weight({config.vocab_size, config.dim}, gen)));
  ASSERT_TRUE(model.set_final_norm(create_random_weight({config.dim}, gen)));
  ASSERT_TRUE(model.set_classifier(create_random_weight({config.vocab_size, config.dim}, gen)));

  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto& block = model.get_block(layer);
    ASSERT_TRUE(block.set_wq(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wk(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wv(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wo(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w1(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w2(create_random_weight({config.dim, config.hidden_dim}, gen)));
    ASSERT_TRUE(block.set_w3(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_attn_norm(create_random_weight({config.dim}, gen)));
    ASSERT_TRUE(block.set_ffn_norm(create_random_weight({config.dim}, gen)));
  }

  // Initialize model (allocates buffers after weights are set)
  ASSERT_TRUE(model.init());

  // Generate next token from prompt [1, 2, 3]
  std::vector<i32> prompt = {1, 2, 3};
  auto next_result = model.generate_next(prompt);
  ASSERT_TRUE(next_result) << next_result.error().message();

  i32 next_token = next_result.value();
  EXPECT_GE(next_token, 0);
  EXPECT_LT(next_token, config.vocab_size);
}

// Test 6: KV cache reset
TEST(LLaMAModelTest, CacheReset) {
  auto config = create_test_config();
  LLaMAModel model(config);

  std::mt19937 gen(789);

  // Initialize weights
  ASSERT_TRUE(model.set_embedding(create_random_weight({config.vocab_size, config.dim}, gen)));
  ASSERT_TRUE(model.set_final_norm(create_random_weight({config.dim}, gen)));
  ASSERT_TRUE(model.set_classifier(create_random_weight({config.vocab_size, config.dim}, gen)));

  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto& block = model.get_block(layer);
    ASSERT_TRUE(block.set_wq(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wk(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wv(create_random_weight({config.kv_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_wo(create_random_weight({config.dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w1(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_w2(create_random_weight({config.dim, config.hidden_dim}, gen)));
    ASSERT_TRUE(block.set_w3(create_random_weight({config.hidden_dim, config.dim}, gen)));
    ASSERT_TRUE(block.set_attn_norm(create_random_weight({config.dim}, gen)));
    ASSERT_TRUE(block.set_ffn_norm(create_random_weight({config.dim}, gen)));
  }

  // Initialize model (allocates buffers after weights are set)
  ASSERT_TRUE(model.init());

  // Generate first sequence
  std::vector<i32> prompt1 = {1, 2};
  auto next1 = model.generate_next(prompt1);
  ASSERT_TRUE(next1);

  // Reset cache
  model.reset_cache();

  // Generate second sequence (should give different result if cache matters)
  std::vector<i32> prompt2 = {3, 4};
  auto next2 = model.generate_next(prompt2);
  ASSERT_TRUE(next2);

  // Both should be valid tokens
  EXPECT_GE(next1.value(), 0);
  EXPECT_LT(next1.value(), config.vocab_size);
  EXPECT_GE(next2.value(), 0);
  EXPECT_LT(next2.value(), config.vocab_size);
}

// Test 7: Argmax sampler
TEST(LLaMAModelTest, ArgmaxSampler) {
  auto logits = Tensor::create({10}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(logits);

  f32* ptr = logits.value().ptr<f32>();
  for (i32 i = 0; i < 10; ++i) {
    ptr[i] = static_cast<f32>(i);
  }
  ptr[7] = 100.0f;  // Max value at index 7

  auto result = argmax_sample(logits.value());
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value(), 7);
}
