/**
 * @file test_error.cpp
 * @brief Unit tests for error.hpp
 */

#include <gtest/gtest.h>

#include "photon/core/error.hpp"

using namespace photon;

// ============================================================================
// Error Tests
// ============================================================================

TEST(ErrorTest, DefaultConstruction) {
  Error err;
  EXPECT_TRUE(err.is_ok());
  EXPECT_FALSE(err.is_err());
  EXPECT_EQ(err.code(), ErrorCode::Ok);
  EXPECT_TRUE(err.message().empty());
}

TEST(ErrorTest, ErrorCodeConstruction) {
  Error err(ErrorCode::OutOfMemory);
  EXPECT_FALSE(err.is_ok());
  EXPECT_TRUE(err.is_err());
  EXPECT_EQ(err.code(), ErrorCode::OutOfMemory);
}

TEST(ErrorTest, ErrorWithMessage) {
  Error err(ErrorCode::FileNotFound, "test.txt not found");
  EXPECT_TRUE(err.is_err());
  EXPECT_EQ(err.code(), ErrorCode::FileNotFound);
  EXPECT_EQ(err.message(), "test.txt not found");
}

TEST(ErrorTest, ToString) {
  Error err1(ErrorCode::OutOfMemory);
  EXPECT_EQ(err1.to_string(), "OutOfMemory");

  Error err2(ErrorCode::FileNotFound, "config.json");
  EXPECT_EQ(err2.to_string(), "FileNotFound: config.json");

  Error ok;
  EXPECT_EQ(ok.to_string(), "Ok");
}

TEST(ErrorTest, BoolConversion) {
  Error ok;
  Error err(ErrorCode::InvalidArgument);

  EXPECT_TRUE(ok);
  EXPECT_FALSE(err);
}

// ============================================================================
// Result Tests
// ============================================================================

TEST(ResultTest, OkValue) {
  Result<int> result = Ok(42);

  EXPECT_TRUE(result.is_ok());
  EXPECT_FALSE(result.is_err());
  EXPECT_TRUE(result);
  EXPECT_EQ(result.value(), 42);
}

TEST(ResultTest, ErrValue) {
  Result<int> result = Err<int>(ErrorCode::InvalidArgument);

  EXPECT_FALSE(result.is_ok());
  EXPECT_TRUE(result.is_err());
  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(ResultTest, ErrWithMessage) {
  Result<int> result = Err<int>(ErrorCode::OutOfMemory, "Failed to allocate");

  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::OutOfMemory);
  EXPECT_EQ(result.error().message(), "Failed to allocate");
}

TEST(ResultTest, ValueOr) {
  Result<int> ok_result = Ok(42);
  Result<int> err_result = Err<int>(ErrorCode::InvalidArgument);

  EXPECT_EQ(ok_result.value_or(0), 42);
  EXPECT_EQ(err_result.value_or(0), 0);
}

TEST(ResultTest, MoveSemantics) {
  Result<std::string> result = Ok(std::string("hello"));

  EXPECT_TRUE(result.is_ok());
  std::string value = std::move(result).value();
  EXPECT_EQ(value, "hello");
}

TEST(ResultTest, VoidResult) {
  Result<void> ok_result = Ok();
  Result<void> err_result = Err<void>(ErrorCode::NotImplemented);

  EXPECT_TRUE(ok_result.is_ok());
  EXPECT_FALSE(ok_result.is_err());
  EXPECT_TRUE(ok_result);

  EXPECT_FALSE(err_result.is_ok());
  EXPECT_TRUE(err_result.is_err());
  EXPECT_FALSE(err_result);
  EXPECT_EQ(err_result.error().code(), ErrorCode::NotImplemented);
}

// ============================================================================
// Practical Usage Tests
// ============================================================================

Result<int> divide(int a, int b) {
  if (b == 0) {
    return Err<int>(ErrorCode::InvalidArgument, "Division by zero");
  }
  return Ok(a / b);
}

TEST(ResultTest, PracticalUsage) {
  auto result1 = divide(10, 2);
  EXPECT_TRUE(result1.is_ok());
  EXPECT_EQ(result1.value(), 5);

  auto result2 = divide(10, 0);
  EXPECT_TRUE(result2.is_err());
  EXPECT_EQ(result2.error().code(), ErrorCode::InvalidArgument);
}

// Test error propagation
Result<int> complex_computation(int x) {
  auto result = divide(x * 2, 2);  // Changed: multiply x by 2 first
  if (!result) {
    return Err<int>(result.error());  // Propagate error
  }
  return Ok(result.value() + 10);
}

TEST(ResultTest, ErrorPropagation) {
  auto result1 = complex_computation(20);
  EXPECT_TRUE(result1.is_ok());
  EXPECT_EQ(result1.value(), 30);  // 20*2/2 + 10 = 30

  auto result2 = complex_computation(0);
  EXPECT_TRUE(result2.is_ok());   // 0*2/2 + 10 = 10, no error!
  EXPECT_EQ(result2.value(), 10);
}
