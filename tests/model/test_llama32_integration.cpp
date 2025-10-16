#include "photon/model/model_loader.hpp"
#include "photon/model/tokenizer.hpp"
#include <gtest/gtest.h>
#include <filesystem>

namespace photon::model::test {

/**
 * @brief Integration tests for real Llama3.2-1B model
 *
 * These tests validate our implementation against the actual model files.
 * Tests are skipped if model files are not found.
 */
class Llama32IntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Model path
    model_dir_ = std::filesystem::path(std::getenv("HOME")) /
                 ".llama/checkpoints/Llama3.2-1B-Instruct";

    model_path_ = model_dir_ / "model.bin";
    tokenizer_path_ = model_dir_ / "tokenizer.model";

    // Check if files exist
    model_exists_ = std::filesystem::exists(model_path_);
    tokenizer_exists_ = std::filesystem::exists(tokenizer_path_);

    if (!model_exists_) {
      GTEST_SKIP() << "Model file not found: " << model_path_;
    }
    if (!tokenizer_exists_) {
      GTEST_SKIP() << "Tokenizer file not found: " << tokenizer_path_;
    }
  }

  std::filesystem::path model_dir_;
  std::filesystem::path model_path_;
  std::filesystem::path tokenizer_path_;
  bool model_exists_ = false;
  bool tokenizer_exists_ = false;
};

// ============================================================================
// Tokenizer Integration Tests
// ============================================================================

TEST_F(Llama32IntegrationTest, LoadRealTokenizer) {
  auto result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(result.is_ok()) << "Failed to load tokenizer: "
                               << result.error().message();

  const auto& tokenizer = result.value();

  // Llama3.2 should have 128256 tokens
  EXPECT_EQ(tokenizer.vocab_size(), 128256);

  // Check special tokens
  EXPECT_EQ(tokenizer.bos_id(), 128000);
  EXPECT_EQ(tokenizer.eos_id(), 128001);
}

TEST_F(Llama32IntegrationTest, EncodeSimpleEnglish) {
  auto result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Test encoding "Hello, world!"
  auto tokens = tokenizer.encode("Hello, world!");

  // Should have multiple tokens
  EXPECT_GT(tokens.size(), 0);
  EXPECT_LT(tokens.size(), 20);  // Reasonable upper bound

  // All tokens should be valid
  for (i32 token : tokens) {
    EXPECT_GE(token, 0);
    EXPECT_LT(token, tokenizer.vocab_size());
  }
}

TEST_F(Llama32IntegrationTest, EncodeDecodRoundTrip) {
  auto result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Test various strings
  std::vector<std::string> test_strings = {
    "Hello",
    "The quick brown fox",
    "1234567890",
    "Hello, world! How are you?",
  };

  for (const auto& original : test_strings) {
    auto tokens = tokenizer.encode(original);
    auto decoded = tokenizer.decode(tokens);

    // Decoded should match original (or be very close)
    EXPECT_FALSE(decoded.empty()) << "Failed to decode: " << original;

    // For simple ASCII, should be exact match
    if (original.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?")
        == std::string::npos) {
      EXPECT_EQ(decoded, original) << "Round-trip mismatch for: " << original;
    }
  }
}

TEST_F(Llama32IntegrationTest, EncodeCommonTokens) {
  auto result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Test that common tokens can be decoded
  std::vector<i32> common_token_ids = {
    0, 1, 2, 3, 4, 5,  // First few tokens
    128000,  // BOS
    128001,  // EOS
  };

  for (i32 token_id : common_token_ids) {
    auto text = tokenizer.decode_token(token_id);
    EXPECT_FALSE(text.empty()) << "Token " << token_id << " decoded to empty string";
  }
}

// ============================================================================
// Model Loader Integration Tests
// ============================================================================

TEST_F(Llama32IntegrationTest, LoadRealModel) {
  auto result = ModelLoader::load(model_path_, false);
  ASSERT_TRUE(result.is_ok()) << "Failed to load model: "
                               << result.error().message();

  const auto& model = result.value();

  // Verify model configuration matches params.json
  EXPECT_EQ(model.config.dim, 2048);
  EXPECT_EQ(model.config.layer_num, 16);
  EXPECT_EQ(model.config.head_num, 32);
  EXPECT_EQ(model.config.kv_head_num, 8);
  EXPECT_EQ(model.config.vocab_size, 128256);

  // Verify computed config
  const auto& tf_config = model.transformer_config;
  EXPECT_EQ(tf_config.dim, 2048);
  EXPECT_EQ(tf_config.head_size, 2048 / 32);  // 64
  EXPECT_EQ(tf_config.kv_dim, (2048 * 8) / 32);  // 512
  EXPECT_EQ(tf_config.kv_mul, 32 / 8);  // 4
}

TEST_F(Llama32IntegrationTest, ModelSizeValidation) {
  auto result = ModelLoader::load(model_path_, false);
  ASSERT_TRUE(result.is_ok());

  const auto& model = result.value();

  // Check that raw_data is valid
  ASSERT_TRUE(model.raw_data != nullptr);
  ASSERT_TRUE(model.raw_data->is_valid());

  // File size should be ~4.7GB
  const usize file_size = model.raw_data->file_size();
  EXPECT_GT(file_size, 4ULL * 1024 * 1024 * 1024);  // > 4GB
  EXPECT_LT(file_size, 5ULL * 1024 * 1024 * 1024);  // < 5GB
}

TEST_F(Llama32IntegrationTest, WeightDataAccess) {
  auto result = ModelLoader::load(model_path_, false);
  ASSERT_TRUE(result.is_ok());

  const auto& model = result.value();

  // Test that we can access weight data
  const void* weight_ptr = model.raw_data->weight(0);
  ASSERT_TRUE(weight_ptr != nullptr);

  // Read first few weights (should be embedding table)
  const f32* weights = static_cast<const f32*>(weight_ptr);

  // Check that weights are reasonable (not NaN, not too large)
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(weights[i] != weights[i]) << "NaN weight at index " << i;  // NaN != NaN
    EXPECT_LT(std::abs(weights[i]), 100.0f) << "Suspiciously large weight at index " << i;
  }
}

// ============================================================================
// End-to-End Integration Test
// ============================================================================

TEST_F(Llama32IntegrationTest, EndToEndModelAndTokenizer) {
  // Load both model and tokenizer
  auto model_result = ModelLoader::load(model_path_, false);
  ASSERT_TRUE(model_result.is_ok());

  auto tokenizer_result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(tokenizer_result.is_ok());

  const auto& model = model_result.value();
  const auto& tokenizer = tokenizer_result.value();

  // Verify vocab sizes match
  EXPECT_EQ(model.config.vocab_size, tokenizer.vocab_size());

  // Test encoding a prompt
  std::string prompt = "Hello, I am a language model";
  auto tokens = tokenizer.encode(prompt);

  EXPECT_GT(tokens.size(), 0);

  // All tokens should be within vocab range
  for (i32 token : tokens) {
    EXPECT_GE(token, 0);
    EXPECT_LT(token, model.config.vocab_size);
  }

  std::cout << "Prompt: \"" << prompt << "\"\n";
  std::cout << "Tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << tokens[i];
  }
  std::cout << "]\n";
  std::cout << "Decoded: \"" << tokenizer.decode(tokens) << "\"\n";
}

TEST_F(Llama32IntegrationTest, TransformerConfigWithTokenizer) {
  // Load model and tokenizer
  auto model_result = ModelLoader::load(model_path_, false);
  ASSERT_TRUE(model_result.is_ok());

  auto tokenizer_result = TikTokenizer::load(tokenizer_path_);
  ASSERT_TRUE(tokenizer_result.is_ok());

  const auto& tokenizer = tokenizer_result.value();
  auto model = std::move(model_result.value());

  // Recompute TransformerConfig with actual tokenizer vocab size
  model.transformer_config = TransformerConfig::from_model_config(
    model.config,
    tokenizer.vocab_size()
  );

  // Should use model's vocab_size since it's >= tokenizer's
  EXPECT_EQ(model.transformer_config.vocab_size, 128256);
  EXPECT_TRUE(model.transformer_config.is_shared_weight);
}

} // namespace photon::model::test
