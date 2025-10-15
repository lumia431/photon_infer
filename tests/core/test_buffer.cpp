/**
 * @file test_buffer.cpp
 * @brief Unit tests for buffer.hpp
 */

#include <gtest/gtest.h>

#include <cstring>

#include "photon/core/buffer.hpp"

using namespace photon;

// ============================================================================
// Buffer Creation Tests
// ============================================================================

TEST(BufferTest, BasicCreation) {
  auto result = Buffer::create(1024, DeviceType::CPU);

  ASSERT_TRUE(result.is_ok());
  Buffer buffer = std::move(result.value());

  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.size(), 1024);
  EXPECT_EQ(buffer.device(), DeviceType::CPU);
  EXPECT_NE(buffer.data(), nullptr);
}

TEST(BufferTest, ZeroSizeCreation) {
  auto result = Buffer::create(0, DeviceType::CPU);

  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(BufferTest, CustomAlignment) {
  usize alignment = 256;
  auto result = Buffer::create(1024, DeviceType::CPU, alignment);

  ASSERT_TRUE(result.is_ok());
  Buffer buffer = std::move(result.value());

  EXPECT_EQ(buffer.alignment(), alignment);

  auto addr = reinterpret_cast<uintptr_t>(buffer.data());
  EXPECT_EQ(addr % alignment, 0);
}

TEST(BufferTest, DefaultConstruction) {
  Buffer buffer;

  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_FALSE(buffer);  // Should convert to false
}

// ============================================================================
// Buffer Move Semantics Tests
// ============================================================================

TEST(BufferTest, MoveConstruction) {
  auto result = Buffer::create(1024, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer buffer1 = std::move(result.value());
  void* ptr = buffer1.data();
  usize size = buffer1.size();

  // Move construct
  Buffer buffer2 = std::move(buffer1);

  EXPECT_TRUE(buffer1.empty());  // buffer1 is now empty
  EXPECT_EQ(buffer1.data(), nullptr);

  EXPECT_FALSE(buffer2.empty());  // buffer2 owns the memory
  EXPECT_EQ(buffer2.data(), ptr);
  EXPECT_EQ(buffer2.size(), size);
}

TEST(BufferTest, MoveAssignment) {
  auto result1 = Buffer::create(1024, DeviceType::CPU);
  auto result2 = Buffer::create(2048, DeviceType::CPU);

  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  Buffer buffer1 = std::move(result1.value());
  Buffer buffer2 = std::move(result2.value());

  void* ptr2 = buffer2.data();

  // Move assign
  buffer1 = std::move(buffer2);

  EXPECT_TRUE(buffer2.empty());
  EXPECT_FALSE(buffer1.empty());
  EXPECT_EQ(buffer1.data(), ptr2);
  EXPECT_EQ(buffer1.size(), 2048);
}

TEST(BufferTest, SelfMoveAssignment) {
  auto result = Buffer::create(1024, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer buffer = std::move(result.value());
  void* ptr = buffer.data();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
  buffer = std::move(buffer);  // Self-assignment
#pragma GCC diagnostic pop

  // Buffer should still be valid
  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.data(), ptr);
}

// ============================================================================
// Buffer Data Access Tests
// ============================================================================

TEST(BufferTest, TypedDataAccess) {
  auto result = Buffer::create(sizeof(f32) * 10, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer buffer = std::move(result.value());

  f32* data = buffer.data_as<f32>();
  EXPECT_NE(data, nullptr);
  EXPECT_EQ(buffer.num_elements<f32>(), 10);

  // Write and read
  for (int i = 0; i < 10; ++i) {
    data[i] = static_cast<f32>(i);
  }

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<f32>(i));
  }
}

TEST(BufferTest, SpanAccess) {
  auto result = Buffer::create(sizeof(i32) * 100, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer buffer = std::move(result.value());

  auto span = buffer.as_span<i32>();
  EXPECT_EQ(span.size(), 100);

  // Write via span
  for (size_t i = 0; i < span.size(); ++i) {
    span[i] = static_cast<i32>(i * 2);
  }

  // Read via span
  for (size_t i = 0; i < span.size(); ++i) {
    EXPECT_EQ(span[i], static_cast<i32>(i * 2));
  }
}

// ============================================================================
// Buffer Operations Tests
// ============================================================================

TEST(BufferTest, Zero) {
  auto result = Buffer::create(1024, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer buffer = std::move(result.value());

  // Fill with non-zero values first
  std::memset(buffer.data(), 0xFF, buffer.size());

  // Zero the buffer
  auto zero_result = buffer.zero();
  EXPECT_TRUE(zero_result.is_ok());

  // Verify all bytes are zero
  auto data = static_cast<const u8*>(buffer.data());
  for (usize i = 0; i < buffer.size(); ++i) {
    EXPECT_EQ(data[i], 0);
  }
}

TEST(BufferTest, CopyFrom) {
  auto result1 = Buffer::create(1024, DeviceType::CPU);
  auto result2 = Buffer::create(1024, DeviceType::CPU);

  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  Buffer src = std::move(result1.value());
  Buffer dst = std::move(result2.value());

  // Fill source with pattern
  auto src_data = src.data_as<u8>();
  for (usize i = 0; i < src.size(); ++i) {
    src_data[i] = static_cast<u8>(i % 256);
  }

  // Copy to destination
  auto copy_result = dst.copy_from(src);
  EXPECT_TRUE(copy_result.is_ok());

  // Verify data
  auto dst_data = dst.data_as<const u8>();
  for (usize i = 0; i < dst.size(); ++i) {
    EXPECT_EQ(dst_data[i], static_cast<u8>(i % 256));
  }
}

TEST(BufferTest, CopyFromSizeMismatch) {
  auto result1 = Buffer::create(1024, DeviceType::CPU);
  auto result2 = Buffer::create(2048, DeviceType::CPU);

  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  Buffer src = std::move(result1.value());
  Buffer dst = std::move(result2.value());

  auto copy_result = dst.copy_from(src);
  EXPECT_TRUE(copy_result.is_err());
  EXPECT_EQ(copy_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(BufferTest, Clone) {
  auto result = Buffer::create(1024, DeviceType::CPU);
  ASSERT_TRUE(result.is_ok());

  Buffer original = std::move(result.value());

  // Fill with pattern
  auto data = original.data_as<u8>();
  for (usize i = 0; i < original.size(); ++i) {
    data[i] = static_cast<u8>(i % 256);
  }

  // Clone
  auto clone_result = original.clone();
  ASSERT_TRUE(clone_result.is_ok());

  Buffer cloned = std::move(clone_result.value());

  // Verify clone has same data
  EXPECT_EQ(cloned.size(), original.size());
  EXPECT_EQ(cloned.device(), original.device());
  EXPECT_NE(cloned.data(), original.data());  // Different memory

  auto cloned_data = cloned.data_as<const u8>();
  for (usize i = 0; i < cloned.size(); ++i) {
    EXPECT_EQ(cloned_data[i], static_cast<u8>(i % 256));
  }
}

TEST(BufferTest, CloneEmpty) {
  Buffer empty;
  auto clone_result = empty.clone();

  ASSERT_TRUE(clone_result.is_ok());
  Buffer cloned = std::move(clone_result.value());

  EXPECT_TRUE(cloned.empty());
}

// ============================================================================
// Buffer Lifetime Tests
// ============================================================================

TEST(BufferTest, RAIIDeallocation) {
  void* ptr = nullptr;

  {
    auto result = Buffer::create(1024, DeviceType::CPU);
    ASSERT_TRUE(result.is_ok());

    Buffer buffer = std::move(result.value());
    ptr = buffer.data();
    EXPECT_NE(ptr, nullptr);

    // buffer goes out of scope here and should deallocate
  }

  // After this point, ptr should be deallocated (we can't verify this
  // directly without memory debugging tools, but the test ensures no crash)
}

// ============================================================================
// Large Buffer Tests
// ============================================================================

TEST(BufferTest, LargeBuffer) {
  usize size = 100 * 1024 * 1024;  // 100 MB
  auto result = Buffer::create(size, DeviceType::CPU);

  ASSERT_TRUE(result.is_ok());
  Buffer buffer = std::move(result.value());

  EXPECT_EQ(buffer.size(), size);

  // Write pattern to verify memory is actually allocated
  auto data = buffer.data_as<u8>();
  for (usize i = 0; i < std::min(size, usize(1000)); ++i) {
    data[i] = static_cast<u8>(i % 256);
  }
}

#ifdef PHOTON_USE_CUDA
// ============================================================================
// CUDA Buffer Tests
// ============================================================================

TEST(BufferTest, CUDACreation) {
  auto result = Buffer::create(1024, DeviceType::CUDA);

  if (result.is_ok()) {
    Buffer buffer = std::move(result.value());

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.device(), DeviceType::CUDA);
    EXPECT_NE(buffer.data(), nullptr);
  } else {
    GTEST_SKIP() << "CUDA not available: " << result.error().to_string();
  }
}

#endif  // PHOTON_USE_CUDA
