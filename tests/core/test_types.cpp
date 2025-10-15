/**
 * @file test_types.cpp
 * @brief Unit tests for types.hpp
 */

#include <gtest/gtest.h>

#include "photon/core/types.hpp"

using namespace photon;

// ============================================================================
// Type Size Tests
// ============================================================================

TEST(TypesTest, IntegerTypeSizes) {
  EXPECT_EQ(sizeof(i8), 1);
  EXPECT_EQ(sizeof(i16), 2);
  EXPECT_EQ(sizeof(i32), 4);
  EXPECT_EQ(sizeof(i64), 8);

  EXPECT_EQ(sizeof(u8), 1);
  EXPECT_EQ(sizeof(u16), 2);
  EXPECT_EQ(sizeof(u32), 4);
  EXPECT_EQ(sizeof(u64), 8);
}

TEST(TypesTest, FloatTypeSizes) {
  EXPECT_EQ(sizeof(f32), 4);
  EXPECT_EQ(sizeof(f64), 8);
}

// ============================================================================
// DataType Tests
// ============================================================================

TEST(TypesTest, DataTypeSize) {
  EXPECT_EQ(data_type_size(DataType::Float32), 4);
  EXPECT_EQ(data_type_size(DataType::Float64), 8);
  EXPECT_EQ(data_type_size(DataType::Int8), 1);
  EXPECT_EQ(data_type_size(DataType::Int16), 2);
  EXPECT_EQ(data_type_size(DataType::Int32), 4);
  EXPECT_EQ(data_type_size(DataType::Int64), 8);
  EXPECT_EQ(data_type_size(DataType::UInt8), 1);
  EXPECT_EQ(data_type_size(DataType::UInt16), 2);
  EXPECT_EQ(data_type_size(DataType::UInt32), 4);
  EXPECT_EQ(data_type_size(DataType::UInt64), 8);
}

TEST(TypesTest, DataTypeString) {
  EXPECT_STREQ(data_type_str(DataType::Float32), "float32");
  EXPECT_STREQ(data_type_str(DataType::Float64), "float64");
  EXPECT_STREQ(data_type_str(DataType::Int32), "int32");
}

TEST(TypesTest, DeviceTypeString) {
  EXPECT_STREQ(device_type_str(DeviceType::CPU), "CPU");
  EXPECT_STREQ(device_type_str(DeviceType::CUDA), "CUDA");
}

// ============================================================================
// Type Mapping Tests
// ============================================================================

TEST(TypesTest, DataTypeToCpp) {
  using F32 = data_type_t<DataType::Float32>;
  using I32 = data_type_t<DataType::Int32>;

  EXPECT_TRUE((std::is_same_v<F32, f32>));
  EXPECT_TRUE((std::is_same_v<I32, i32>));
}

TEST(TypesTest, CppToDataType) {
  EXPECT_EQ(cpp_type_to_data_type_v<f32>, DataType::Float32);
  EXPECT_EQ(cpp_type_to_data_type_v<f64>, DataType::Float64);
  EXPECT_EQ(cpp_type_to_data_type_v<i32>, DataType::Int32);
  EXPECT_EQ(cpp_type_to_data_type_v<u8>, DataType::UInt8);
}

// ============================================================================
// Concept Tests
// ============================================================================

// Test that concepts work at compile time
static_assert(FloatingPoint<f32>, "f32 should satisfy FloatingPoint");
static_assert(FloatingPoint<f64>, "f64 should satisfy FloatingPoint");
static_assert(!FloatingPoint<i32>, "i32 should not satisfy FloatingPoint");

static_assert(Integral<i32>, "i32 should satisfy Integral");
static_assert(Integral<u32>, "u32 should satisfy Integral");
static_assert(!Integral<f32>, "f32 should not satisfy Integral");

static_assert(Numeric<f32>, "f32 should satisfy Numeric");
static_assert(Numeric<i32>, "i32 should satisfy Numeric");

static_assert(SignedNumeric<i32>, "i32 should satisfy SignedNumeric");
static_assert(SignedNumeric<f32>, "f32 should satisfy SignedNumeric");
static_assert(!SignedNumeric<u32>, "u32 should not satisfy SignedNumeric");

static_assert(UnsignedNumeric<u32>, "u32 should satisfy UnsignedNumeric");
static_assert(!UnsignedNumeric<i32>, "i32 should not satisfy UnsignedNumeric");
static_assert(!UnsignedNumeric<f32>, "f32 should not satisfy UnsignedNumeric");
