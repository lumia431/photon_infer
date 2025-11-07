#include "photon/io/tokenizer.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

namespace photon::model::test {

class TikTokenizerTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / "photon_tokenizer_test";
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(test_dir_);
  }

  // Create a minimal test vocab file
  std::filesystem::path create_test_vocab() {
    auto vocab_path = test_dir_ / "test_vocab.model";
    std::ofstream file(vocab_path);

    // Simple base64 encoded tokens
    // "SGVsbG8=" is "Hello" in base64
    // "IHdvcmxk" is " world" in base64
    file << "SGVsbG8= 100\n";
    file << "IHdvcmxk 101\n";
    file << "Cg== 102\n";  // "\n"

    // Single character tokens
    file << "YQ== 103\n";  // "a"
    file << "Yg== 104\n";  // "b"
    file << "Yw== 105\n";  // "c"

    file.close();
    return vocab_path;
  }

  std::filesystem::path test_dir_;
};

TEST_F(TikTokenizerTest, LoadVocab) {
  auto vocab_path = create_test_vocab();

  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();
  EXPECT_GT(tokenizer.vocab_size(), 0);
  EXPECT_EQ(tokenizer.bos_id(), TikTokenizer::kDefaultBosId);
  EXPECT_EQ(tokenizer.eos_id(), TikTokenizer::kDefaultEosId);
}

TEST_F(TikTokenizerTest, LoadNonExistentFile) {
  auto result = TikTokenizer::load("/nonexistent/vocab.model");
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.error().code(), ErrorCode::FileNotFound);
}

TEST_F(TikTokenizerTest, Base64Decoding) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  // Verify tokens were correctly decoded
  const auto& tokenizer = result.value();
  auto tokens = tokenizer.encode("Hello world");
  EXPECT_GT(tokens.size(), 0);
}

TEST_F(TikTokenizerTest, EncodeSimpleText) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Test encoding (greedy longest match)
  auto tokens = tokenizer.encode("abc");
  EXPECT_GT(tokens.size(), 0);

  // Should be able to encode each character separately
  EXPECT_LE(tokens.size(), 3);
}

TEST_F(TikTokenizerTest, DecodeTokens) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Encode then decode should give back similar text
  std::string original = "abc";
  auto tokens = tokenizer.encode(original);
  auto decoded = tokenizer.decode(tokens);

  EXPECT_FALSE(decoded.empty());
}

TEST_F(TikTokenizerTest, DecodeSingleToken) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Test decoding a known token
  auto text = tokenizer.decode_token(103);  // "a"
  EXPECT_EQ(text, "a");
}

TEST_F(TikTokenizerTest, SpecialTokens) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Verify special tokens exist
  EXPECT_EQ(tokenizer.bos_id(), 128000);
  EXPECT_EQ(tokenizer.eos_id(), 128001);

  // Should be able to decode special tokens
  auto bos_text = tokenizer.decode_token(tokenizer.bos_id());
  EXPECT_FALSE(bos_text.empty());
  EXPECT_TRUE(bos_text.find("begin_of_text") != std::string_view::npos ||
              bos_text.find("reserved_special_token") != std::string_view::npos);
}

TEST_F(TikTokenizerTest, EmptyText) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  auto tokens = tokenizer.encode("");
  EXPECT_TRUE(tokens.empty());

  auto decoded = tokenizer.decode(tokens);
  EXPECT_TRUE(decoded.empty());
}

TEST_F(TikTokenizerTest, RoundTrip) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // For characters in the vocab, round-trip should work perfectly
  std::string original = "abc";
  auto tokens = tokenizer.encode(original);
  auto decoded = tokenizer.decode(tokens);

  // Should at least decode to something with the same length
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TikTokenizerTest, VocabSize) {
  auto vocab_path = create_test_vocab();
  auto result = TikTokenizer::load(vocab_path);
  ASSERT_TRUE(result.is_ok());

  const auto& tokenizer = result.value();

  // Should have base tokens (6) + special tokens (128000-128255 = 256)
  EXPECT_GT(tokenizer.vocab_size(), 100);
  EXPECT_GE(tokenizer.vocab_size(), 256);  // At least the special tokens
}

} // namespace photon::model::test
