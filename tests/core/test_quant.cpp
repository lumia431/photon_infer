/**
 * @file test_quant.cpp
 * @brief Unit tests for int8 quantization functionality
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "photon/core/quant.hpp"
#include "photon/core/tensor.hpp"

namespace photon::test {

// ============================================================================
// Test Fixtures
// ============================================================================

class QuantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Seed for reproducibility
    rng_.seed(42);
  }

  // Generate random float32 weights
  std::vector<f32> generate_weights(usize size, f32 min = -1.0f, f32 max = 1.0f) {
    std::uniform_real_distribution<f32> dist(min, max);
    std::vector<f32> weights(size);
    for (auto& w : weights) {
      w = dist(rng_);
    }
    return weights;
  }

  std::mt19937 rng_;
};

// ============================================================================
// Basic Quantization Tests
// ============================================================================

TEST_F(QuantTest, QuantizeValue) {
  // Test quantize_value function
  f32 scale = 0.1f;

  // Test positive value
  i8 q1 = quantize_value(5.0f, scale);
  EXPECT_EQ(q1, 50);

  // Test negative value
  i8 q2 = quantize_value(-5.0f, scale);
  EXPECT_EQ(q2, -50);

  // Test clamping (should clamp to [-127, 127])
  i8 q3 = quantize_value(20.0f, scale);  // Would be 200, clamped to 127
  EXPECT_EQ(q3, 127);

  i8 q4 = quantize_value(-20.0f, scale);  // Would be -200, clamped to -127
  EXPECT_EQ(q4, -127);
}

TEST_F(QuantTest, DequantizeValue) {
  f32 scale = 0.1f;

  f32 dq1 = dequantize_value(50, scale);
  EXPECT_FLOAT_EQ(dq1, 5.0f);

  f32 dq2 = dequantize_value(-50, scale);
  EXPECT_FLOAT_EQ(dq2, -5.0f);
}

TEST_F(QuantTest, ComputeAbsmax) {
  std::vector<f32> data = {-3.0f, 1.0f, 5.0f, -2.0f};
  f32 absmax = compute_absmax(data);
  EXPECT_FLOAT_EQ(absmax, 5.0f);
}

TEST_F(QuantTest, ComputeScale) {
  f32 max_abs = 12.7f;
  f32 scale = compute_scale(max_abs);
  EXPECT_FLOAT_EQ(scale, 12.7f / 127.0f);
}

// ============================================================================
// Group-wise Quantization Tests
// ============================================================================

TEST_F(QuantTest, QuantizeWeightsSmall) {
  // Small test with known values
  std::vector<f32> weights = {1.0f, 2.0f, 3.0f, 4.0f};
  i32 group_size = 4;  // Single group

  auto result = quantize_weights(weights, group_size);
  ASSERT_TRUE(result);

  auto [quantized, params] = std::move(result.value());

  // Check parameters
  EXPECT_EQ(params.group_size, 4);
  EXPECT_EQ(params.num_groups, 1);
  EXPECT_EQ(params.scales.size(), 1);

  // Check scale is reasonable
  f32 expected_scale = 4.0f / 127.0f;
  EXPECT_NEAR(params.scales[0], expected_scale, 1e-5f);

  // Check quantized values
  EXPECT_EQ(quantized.size(), 4);
}

TEST_F(QuantTest, QuantizeWeightsMultipleGroups) {
  // Test with multiple groups
  std::vector<f32> weights(256);
  for (usize i = 0; i < 256; ++i) {
    weights[i] = static_cast<f32>(i) / 10.0f;
  }

  i32 group_size = 128;

  auto result = quantize_weights(weights, group_size);
  ASSERT_TRUE(result);

  auto [quantized, params] = std::move(result.value());

  // Check parameters
  EXPECT_EQ(params.group_size, 128);
  EXPECT_EQ(params.num_groups, 2);
  EXPECT_EQ(params.scales.size(), 2);

  // Check quantized size
  EXPECT_EQ(quantized.size(), 256);
}

TEST_F(QuantTest, QuantizeDequantizeRoundtrip) {
  // Generate random weights
  auto weights = generate_weights(512, -10.0f, 10.0f);
  i32 group_size = 128;

  // Quantize
  auto quant_result = quantize_weights(weights, group_size);
  ASSERT_TRUE(quant_result);
  auto [quantized, params] = std::move(quant_result.value());

  // Dequantize
  auto dequant_result = dequantize_weights(quantized, params);
  ASSERT_TRUE(dequant_result);
  auto dequantized = std::move(dequant_result.value());

  // Check size
  EXPECT_EQ(dequantized.size(), weights.size());

  // Check error is small
  f32 max_error = 0.0f;
  for (usize i = 0; i < weights.size(); ++i) {
    f32 error = std::abs(weights[i] - dequantized[i]);
    max_error = std::max(max_error, error);
  }

  // Error should be less than 2 * scale (worst case rounding)
  f32 max_scale = *std::max_element(params.scales.begin(), params.scales.end());
  EXPECT_LT(max_error, 2.0f * max_scale);
}

// ============================================================================
// Quantization Error Tests
// ============================================================================

TEST_F(QuantTest, QuantizationError) {
  auto weights = generate_weights(1024, -5.0f, 5.0f);
  i32 group_size = 128;

  auto quant_result = quantize_weights(weights, group_size);
  ASSERT_TRUE(quant_result);
  auto [quantized, params] = std::move(quant_result.value());

  // Compute error
  auto dequant_result = dequantize_weights(quantized, params);
  ASSERT_TRUE(dequant_result);
  auto dequantized = std::move(dequant_result.value());

  f32 rmse = compute_quantization_error(weights, dequantized);

  // RMSE should be small (< 1% of range)
  EXPECT_LT(rmse, 0.1f);  // 10.0f range, so < 1%
}

TEST_F(QuantTest, QuantizationStats) {
  auto weights = generate_weights(512, -3.0f, 3.0f);
  i32 group_size = 128;

  auto quant_result = quantize_weights(weights, group_size);
  ASSERT_TRUE(quant_result);
  auto [quantized, params] = std::move(quant_result.value());

  // Compute stats
  auto stats = compute_quant_stats(weights, quantized, params);

  // Check stats are reasonable
  EXPECT_EQ(stats.num_groups, 4);
  EXPECT_GT(stats.compression_ratio, 3.0f);  // Should be ~3.5x
  EXPECT_LT(stats.rmse, 0.1f);

  std::cout << "Quantization Stats:\n";
  std::cout << stats.to_string() << std::endl;
}

// ============================================================================
// Tensor Quantization Tests
// ============================================================================

TEST_F(QuantTest, QuantizeTensor) {
  // Create a float32 tensor
  auto weights_data = generate_weights(256, -1.0f, 1.0f);
  auto tensor_result = Tensor::from_matrix(16, 16, weights_data, DeviceType::CPU);
  ASSERT_TRUE(tensor_result);
  auto tensor = std::move(tensor_result.value());

  // Quantize
  i32 group_size = 64;
  auto quant_result = quantize_tensor(tensor, group_size);
  ASSERT_TRUE(quant_result);

  auto [quant_tensor, params] = std::move(quant_result.value());

  // Check quantized tensor properties
  EXPECT_EQ(quant_tensor.dtype(), DataType::Int8);
  EXPECT_EQ(quant_tensor.ndim(), 2);
  EXPECT_EQ(quant_tensor.dim(0), 16);
  EXPECT_EQ(quant_tensor.dim(1), 16);
  EXPECT_EQ(quant_tensor.size(), 256);

  // Check params
  EXPECT_EQ(params.group_size, 64);
  EXPECT_EQ(params.num_groups, 4);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(QuantTest, AllZeroWeights) {
  std::vector<f32> weights(128, 0.0f);
  i32 group_size = 64;

  auto result = quantize_weights(weights, group_size);
  ASSERT_TRUE(result);

  auto [quantized, params] = std::move(result.value());

  // All quantized values should be 0
  for (auto q : quantized) {
    EXPECT_EQ(q, 0);
  }
}

TEST_F(QuantTest, SmallGroupSize) {
  auto weights = generate_weights(128, -1.0f, 1.0f);
  i32 group_size = 8;  // Small group size

  auto result = quantize_weights(weights, group_size);
  ASSERT_TRUE(result);

  auto [quantized, params] = std::move(result.value());

  EXPECT_EQ(params.num_groups, 16);
  EXPECT_EQ(params.scales.size(), 16);
}

TEST_F(QuantTest, LargeGroupSize) {
  auto weights = generate_weights(256, -1.0f, 1.0f);
  i32 group_size = 256;  // Single group

  auto result = quantize_weights(weights, group_size);
  ASSERT_TRUE(result);

  auto [quantized, params] = std::move(result.value());

  EXPECT_EQ(params.num_groups, 1);
  EXPECT_EQ(params.scales.size(), 1);
}

}  // namespace photon::test
