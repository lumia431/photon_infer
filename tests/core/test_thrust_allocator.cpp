/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file test_thrust_allocator.cpp
 * @brief Comprehensive unit tests for ThrustAllocator
 */

#include "photon/core/thrust_allocator.hpp"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

namespace photon {
namespace test {

class ThrustAllocatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    allocator_ = std::make_unique<ThrustAllocator>();
  }

  void TearDown() override {
    allocator_.reset();
    cudaDeviceReset();
  }

  std::unique_ptr<ThrustAllocator> allocator_;
};

// Test 1: Basic allocation and deallocation
TEST_F(ThrustAllocatorTest, BasicAllocation) {
  constexpr usize size = 1024 * 1024;

  auto result = allocator_->allocate(size);
  ASSERT_TRUE(result.is_ok());

  void* ptr = std::move(result).unwrap();
  ASSERT_NE(ptr, nullptr);

  auto stats = allocator_->get_stats();
  EXPECT_EQ(stats.total_allocated, size);
  EXPECT_EQ(stats.current_usage, size);
  EXPECT_EQ(stats.peak_usage, size);

  auto dealloc_result = allocator_->deallocate(ptr, size);
  ASSERT_TRUE(dealloc_result.is_ok());

  stats = allocator_->get_stats();
  EXPECT_EQ(stats.total_freed, size);
  EXPECT_EQ(stats.current_usage, 0);
}

// Test 2: Repeated cycles (simulating leak scenario)
TEST_F(ThrustAllocatorTest, RepeatedCycles) {
  constexpr int num_cycles = 10;
  constexpr usize size = 424 * 1024 * 1024; // 424MB

  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    auto result = allocator_->allocate(size);
    ASSERT_TRUE(result.is_ok());
    void* ptr = std::move(result).unwrap();

    auto stats = allocator_->get_stats();
    EXPECT_EQ(stats.current_usage, size);

    auto dealloc_result = allocator_->deallocate(ptr, size);
    ASSERT_TRUE(dealloc_result.is_ok());

    stats = allocator_->get_stats();
    EXPECT_EQ(stats.current_usage, 0)
      << "Cycle " << cycle << ": Memory leak detected!";
  }

  auto final_stats = allocator_->get_stats();
  EXPECT_EQ(final_stats.total_allocated, final_stats.total_freed);
  EXPECT_EQ(final_stats.current_usage, 0);
}

// Test 3: Multiple allocations
TEST_F(ThrustAllocatorTest, MultipleAllocations) {
  constexpr int num_allocs = 10;
  constexpr usize size = 1024 * 1024;

  std::vector<void*> ptrs;
  for (int i = 0; i < num_allocs; ++i) {
    auto result = allocator_->allocate(size);
    ASSERT_TRUE(result.is_ok());
    ptrs.push_back(std::move(result).unwrap());
  }

  auto stats = allocator_->get_stats();
  EXPECT_EQ(stats.total_allocated, size * num_allocs);
  EXPECT_EQ(stats.current_usage, size * num_allocs);

  for (void* ptr : ptrs) {
    auto res = allocator_->deallocate(ptr, size);
    ASSERT_TRUE(res.is_ok());
  }

  stats = allocator_->get_stats();
  EXPECT_EQ(stats.current_usage, 0);
}

// Test 4: Zero-size allocation
TEST_F(ThrustAllocatorTest, ZeroSizeAllocation) {
  auto result = allocator_->allocate(0);
  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

// Test 5: Null pointer deallocation
TEST_F(ThrustAllocatorTest, NullPointerDeallocation) {
  auto result = allocator_->deallocate(nullptr, 0);
  ASSERT_TRUE(result.is_ok());
}

// Test 6: Mixed size allocations
TEST_F(ThrustAllocatorTest, MixedSizeAllocations) {
  std::vector<std::pair<void*, usize>> allocations;
  std::vector<usize> sizes = {
    1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024
  };

  usize total_size = 0;
  for (usize size : sizes) {
    auto result = allocator_->allocate(size);
    ASSERT_TRUE(result.is_ok());
    allocations.emplace_back(std::move(result).unwrap(), size);
    total_size += size;
  }

  auto stats = allocator_->get_stats();
  EXPECT_EQ(stats.current_usage, total_size);

  for (auto& [ptr, size] : allocations) {
    auto res = allocator_->deallocate(ptr, size);
    ASSERT_TRUE(res.is_ok());
  }

  stats = allocator_->get_stats();
  EXPECT_EQ(stats.current_usage, 0);
}

// Test 7: Peak usage tracking
TEST_F(ThrustAllocatorTest, PeakUsageTracking) {
  constexpr usize size1 = 100 * 1024 * 1024;
  constexpr usize size2 = 200 * 1024 * 1024;

  auto result1 = allocator_->allocate(size1);
  ASSERT_TRUE(result1.is_ok());
  void* ptr1 = std::move(result1).unwrap();

  auto stats = allocator_->get_stats();
  EXPECT_EQ(stats.peak_usage, size1);

  auto result2 = allocator_->allocate(size2);
  ASSERT_TRUE(result2.is_ok());
  void* ptr2 = std::move(result2).unwrap();

  stats = allocator_->get_stats();
  EXPECT_EQ(stats.current_usage, size1 + size2);
  EXPECT_EQ(stats.peak_usage, size1 + size2);

  auto res1 = allocator_->deallocate(ptr1, size1);
  ASSERT_TRUE(res1.is_ok());

  stats = allocator_->get_stats();
  EXPECT_EQ(stats.current_usage, size2);
  EXPECT_EQ(stats.peak_usage, size1 + size2); // Peak should not decrease

  auto res2 = allocator_->deallocate(ptr2, size2);
  ASSERT_TRUE(res2.is_ok());
}

} // namespace test
} // namespace photon

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
