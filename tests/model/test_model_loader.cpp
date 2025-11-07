#include "photon/io/model_loader.hpp"
#include "photon/arch/config.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

namespace photon::model::test {

class ModelLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / "photon_model_test";
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(test_dir_);
  }

  // Create a minimal test model file
  std::filesystem::path create_test_model(bool is_quantized = false) {
    auto model_path = test_dir_ / (is_quantized ? "test_quant.bin" : "test_fp32.bin");

    std::ofstream file(model_path, std::ios::binary);
    EXPECT_TRUE(file.is_open());

    // Write ModelConfig
    ModelConfig config;
    config.dim = 512;
    config.hidden_dim = 2048;
    config.layer_num = 4;
    config.head_num = 8;
    config.kv_head_num = 4;
    config.vocab_size = is_quantized ? -32000 : 32000; // negative = not shared
    config.seq_len = 2048;

    file.write(reinterpret_cast<const char*>(&config), sizeof(ModelConfig));

    // Write group size for quantized models
    if (is_quantized) {
      i32 group_size = 64;
      file.write(reinterpret_cast<const char*>(&group_size), sizeof(i32));
    }

    // Write some dummy weights (100 floats or int8s)
    if (is_quantized) {
      std::vector<i8> weights(100, 42);
      file.write(reinterpret_cast<const char*>(weights.data()), weights.size());
    } else {
      std::vector<f32> weights(100, 3.14f);
      file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(f32));
    }

    file.close();
    return model_path;
  }

  std::filesystem::path test_dir_;
};

TEST_F(ModelLoaderTest, LoadFp32Model) {
  auto model_path = create_test_model(false);

  auto result = ModelLoader::load(model_path, false);
  ASSERT_TRUE(result.is_ok());

  const auto& loaded = result.value();
  EXPECT_EQ(loaded.config.dim, 512);
  EXPECT_EQ(loaded.config.hidden_dim, 2048);
  EXPECT_EQ(loaded.config.layer_num, 4);
  EXPECT_EQ(loaded.config.head_num, 8);
  EXPECT_EQ(loaded.config.kv_head_num, 4);
  EXPECT_EQ(loaded.config.seq_len, 2048);
  EXPECT_FALSE(loaded.is_quantized);
  EXPECT_EQ(loaded.group_size, 1);
  EXPECT_TRUE(loaded.raw_data != nullptr);
  EXPECT_TRUE(loaded.raw_data->is_valid());
}

TEST_F(ModelLoaderTest, LoadQuantizedModel) {
  auto model_path = create_test_model(true);

  auto result = ModelLoader::load(model_path, true);
  ASSERT_TRUE(result.is_ok());

  const auto& loaded = result.value();
  EXPECT_EQ(loaded.config.dim, 512);
  EXPECT_TRUE(loaded.is_quantized);
  EXPECT_EQ(loaded.group_size, 64);
  EXPECT_TRUE(loaded.raw_data != nullptr);
  EXPECT_TRUE(loaded.raw_data->is_valid());
}

TEST_F(ModelLoaderTest, LoadNonExistentFile) {
  auto result = ModelLoader::load("/nonexistent/path/model.bin", false);
  ASSERT_FALSE(result.is_ok());
  EXPECT_EQ(result.error().code(), ErrorCode::FileNotFound);
}

TEST_F(ModelLoaderTest, TransformerConfigComputation) {
  auto model_path = create_test_model(false);
  auto result = ModelLoader::load(model_path, false);
  ASSERT_TRUE(result.is_ok());

  const auto& tf_config = result.value().transformer_config;
  EXPECT_EQ(tf_config.dim, 512);
  EXPECT_EQ(tf_config.head_size, 512 / 8);  // dim / head_num
  EXPECT_EQ(tf_config.kv_dim, 512 * 4 / 8);  // (dim * kv_head_num) / head_num
  EXPECT_EQ(tf_config.kv_mul, 8 / 4);  // head_num / kv_head_num
}

TEST_F(ModelLoaderTest, TypedLoadFp32) {
  auto model_path = create_test_model(false);

  auto result = ModelLoader::load_typed<f32>(model_path);
  ASSERT_TRUE(result.is_ok());

  const auto& loaded = result.value();
  EXPECT_EQ(loaded.config.dim, 512);
  EXPECT_TRUE(loaded.raw_data != nullptr);

  // Verify we can access weights
  const void* weight_ptr = loaded.raw_data->weight(0);
  EXPECT_TRUE(weight_ptr != nullptr);
}

TEST_F(ModelLoaderTest, TypedLoadInt8) {
  auto model_path = create_test_model(true);

  auto result = ModelLoader::load_typed<i8>(model_path);
  ASSERT_TRUE(result.is_ok());

  const auto& loaded = result.value();
  EXPECT_EQ(loaded.config.dim, 512);
  EXPECT_EQ(loaded.group_size, 64);
  EXPECT_TRUE(loaded.raw_data != nullptr);
}

TEST_F(ModelLoaderTest, WeightAccess) {
  auto model_path = create_test_model(false);
  auto result = ModelLoader::load(model_path, false);
  ASSERT_TRUE(result.is_ok());

  const auto& loaded = result.value();
  // Access weight at offset 0
  const void* weight0 = loaded.raw_data->weight(0);
  EXPECT_TRUE(weight0 != nullptr);

  // Access weight at offset 10
  const void* weight10 = loaded.raw_data->weight(10);
  EXPECT_TRUE(weight10 != nullptr);

  // For FP32, offsets should be 40 bytes apart (10 * sizeof(float))
  const f32* fp32_0 = static_cast<const f32*>(weight0);
  const f32* fp32_10 = static_cast<const f32*>(weight10);
  EXPECT_EQ(fp32_10 - fp32_0, 10);
}

TEST_F(ModelLoaderTest, SharedWeights) {
  auto model_path = test_dir_ / "shared.bin";

  std::ofstream file(model_path, std::ios::binary);
  ModelConfig config;
  config.dim = 256;
  config.hidden_dim = 1024;
  config.layer_num = 2;
  config.head_num = 4;
  config.kv_head_num = 2;
  config.vocab_size = 32000;  // Positive = shared weights
  config.seq_len = 1024;

  file.write(reinterpret_cast<const char*>(&config), sizeof(ModelConfig));
  std::vector<f32> weights(10, 1.0f);
  file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(f32));
  file.close();

  auto result = ModelLoader::load(model_path, false);
  ASSERT_TRUE(result.is_ok());
  EXPECT_TRUE(result.value().transformer_config.is_shared_weight);
}

} // namespace photon::model::test
