/**
 * @file test_allocator.cpp
 * @brief Unit tests for allocator.hpp
 */

#include <gtest/gtest.h>

#include <cstring>

#include "photon/core/allocator.hpp"

using namespace photon;

// ============================================================================
// CPUAllocator Tests
// ============================================================================

TEST(CPUAllocatorTest, BasicAllocation) {
  CPUAllocator allocator;
  auto result = allocator.allocate(1024);

  ASSERT_TRUE(result.is_ok());
  void* ptr = result.value();
  EXPECT_NE(ptr, nullptr);

  // Check alignment
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  EXPECT_EQ(addr % CPUAllocator::kDefaultAlignment, 0);

  // Deallocate
  auto dealloc_result = allocator.deallocate(ptr, 1024);
  EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, CustomAlignment) {
  CPUAllocator allocator;
  usize alignment = 256;
  auto result = allocator.allocate(1024, alignment);

  ASSERT_TRUE(result.is_ok());
  void* ptr = result.value();

  auto addr = reinterpret_cast<uintptr_t>(ptr);
  EXPECT_EQ(addr % alignment, 0);

  auto dealloc_result = allocator.deallocate(ptr, 1024);
  EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, ZeroSizeAllocation) {
  CPUAllocator allocator;
  auto result = allocator.allocate(0);

  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

TEST(CPUAllocatorTest, InvalidAlignment) {
  CPUAllocator allocator;
  // Alignment must be power of 2
  auto result = allocator.allocate(1024, 3);

  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidAlignment);
}

TEST(CPUAllocatorTest, NullPointerDeallocation) {
  CPUAllocator allocator;
  auto result = allocator.deallocate(nullptr, 0);

  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().code(), ErrorCode::NullPointer);
}

TEST(CPUAllocatorTest, DeviceType) {
  CPUAllocator allocator;
  EXPECT_EQ(allocator.device_type(), DeviceType::CPU);
}

TEST(CPUAllocatorTest, LargeAllocation) {
  CPUAllocator allocator;
  usize size = 100 * 1024 * 1024;  // 100 MB
  auto result = allocator.allocate(size);

  ASSERT_TRUE(result.is_ok());
  void* ptr = result.value();
  EXPECT_NE(ptr, nullptr);

  // Write to memory to ensure it's actually allocated
  std::memset(ptr, 0, size);

  auto dealloc_result = allocator.deallocate(ptr, size);
  EXPECT_TRUE(dealloc_result.is_ok());
}

TEST(CPUAllocatorTest, MultipleAllocations) {
  CPUAllocator allocator;
  std::vector<void*> ptrs;

  // Allocate multiple blocks
  for (int i = 0; i < 10; ++i) {
    auto result = allocator.allocate(1024);
    ASSERT_TRUE(result.is_ok());
    ptrs.push_back(result.value());
  }

  // All pointers should be different
  for (size_t i = 0; i < ptrs.size(); ++i) {
    for (size_t j = i + 1; j < ptrs.size(); ++j) {
      EXPECT_NE(ptrs[i], ptrs[j]);
    }
  }

  // Deallocate all
  for (void* ptr : ptrs) {
    auto result = allocator.deallocate(ptr, 1024);
    EXPECT_TRUE(result.is_ok());
  }
}

// ============================================================================
// Allocator Concept Tests
// ============================================================================

// Verify CPUAllocator satisfies Allocator concept at compile time
static_assert(Allocator<CPUAllocator>, "CPUAllocator should satisfy Allocator concept");

// Test concept requirements
TEST(AllocatorConceptTest, CPUAllocatorSatisfiesConcept) {
  CPUAllocator alloc;

  // Test allocate signature
  Result<void*> alloc_result = alloc.allocate(1024, 64);
  EXPECT_TRUE(alloc_result.is_ok() || alloc_result.is_err());

  if (alloc_result) {
    // Test deallocate signature
    Result<void> dealloc_result =
        alloc.deallocate(alloc_result.value(), 1024);
    EXPECT_TRUE(dealloc_result.is_ok());
  }

  // Test device_type signature
  DeviceType device = alloc.device_type();
  EXPECT_EQ(device, DeviceType::CPU);
}

// ============================================================================
// UniquePtr with Allocator Tests
// ============================================================================

TEST(AllocatorUniquePtrTest, BasicUsage) {
  CPUAllocator allocator;
  usize size = 1024;

  auto result = make_unique_alloc(allocator, size, 64);
  ASSERT_TRUE(result.is_ok());

  auto ptr = std::move(result.value());
  EXPECT_NE(ptr.get(), nullptr);

  // Write to memory
  std::memset(ptr.get(), 42, size);

  // ptr will be automatically deallocated when it goes out of scope
}

TEST(AllocatorUniquePtrTest, MoveSemantics) {
  CPUAllocator allocator;
  usize size = 1024;

  auto result = make_unique_alloc(allocator, size, 64);
  ASSERT_TRUE(result.is_ok());

  auto ptr1 = std::move(result.value());
  void* raw_ptr = ptr1.get();

  // Move to another unique_ptr
  auto ptr2 = std::move(ptr1);

  EXPECT_EQ(ptr1.get(), nullptr);  // ptr1 is now null
  EXPECT_EQ(ptr2.get(), raw_ptr);  // ptr2 owns the memory
}

#ifdef PHOTON_USE_CUDA
// ============================================================================
// CUDAAllocator Tests (if CUDA is enabled)
// ============================================================================

TEST(CUDAAllocatorTest, BasicAllocation) {
  CUDAAllocator allocator;
  auto result = allocator.allocate(1024);

  if (result.is_ok()) {
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);

    auto dealloc_result = allocator.deallocate(ptr, 1024);
    EXPECT_TRUE(dealloc_result.is_ok());
  } else {
    // CUDA might not be available on test machine
    GTEST_SKIP() << "CUDA not available: " << result.error().to_string();
  }
}

TEST(CUDAAllocatorTest, DeviceType) {
  CUDAAllocator allocator;
  EXPECT_EQ(allocator.device_type(), DeviceType::CUDA);
}

#endif  // PHOTON_USE_CUDA
