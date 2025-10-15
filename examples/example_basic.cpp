/**
 * @file example_basic.cpp
 * @brief Basic usage examples for PhotonInfer PR1 features
 *
 * This file demonstrates the core features introduced in PR1:
 * - Result<T> error handling
 * - Memory allocators
 * - Buffer management
 * - C++20 Concepts
 */

#include <iostream>

#include "photon/core/allocator.hpp"
#include "photon/core/buffer.hpp"
#include "photon/core/error.hpp"
#include "photon/core/types.hpp"

using namespace photon;

// ============================================================================
// Example 1: Error Handling with Result<T>
// ============================================================================

Result<f32> safe_divide(f32 a, f32 b) {
  if (b == 0.0f) {
    return Err<f32>(ErrorCode::InvalidArgument, "Division by zero");
  }
  return Ok(a / b);
}

void example_error_handling() {
  std::cout << "\n=== Example 1: Error Handling ===\n";

  // Success case
  auto result1 = safe_divide(10.0f, 2.0f);
  if (result1) {
    std::cout << "10.0 / 2.0 = " << result1.value() << "\n";
  }

  // Error case
  auto result2 = safe_divide(10.0f, 0.0f);
  if (!result2) {
    std::cout << "Error: " << result2.error().to_string() << "\n";
  }

  // Using value_or
  f32 safe_result = safe_divide(10.0f, 0.0f).value_or(-1.0f);
  std::cout << "Safe result with default: " << safe_result << "\n";
}

// ============================================================================
// Example 2: Memory Allocation
// ============================================================================

void example_allocator() {
  std::cout << "\n=== Example 2: Memory Allocator ===\n";

  CPUAllocator allocator;

  // Allocate aligned memory
  auto result = allocator.allocate(1024, 64);
  if (!result) {
    std::cerr << "Allocation failed: " << result.error().to_string() << "\n";
    return;
  }

  void* ptr = result.value();
  std::cout << "Allocated 1024 bytes at address: " << ptr << "\n";
  std::cout << "Alignment check: "
            << (reinterpret_cast<uintptr_t>(ptr) % 64 == 0 ? "OK" : "FAIL")
            << "\n";

  // Use the memory
  f32* data = static_cast<f32*>(ptr);
  for (int i = 0; i < 10; ++i) {
    data[i] = static_cast<f32>(i * 2);
  }

  std::cout << "Written data: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "\n";

  // Deallocate
  auto dealloc_result = allocator.deallocate(ptr, 1024);
  if (dealloc_result) {
    std::cout << "Memory deallocated successfully\n";
  }
}

// ============================================================================
// Example 3: Buffer Management
// ============================================================================

void example_buffer() {
  std::cout << "\n=== Example 3: Buffer Management ===\n";

  // Create a buffer
  auto buffer_result = Buffer::create(sizeof(f32) * 100, DeviceType::CPU);
  if (!buffer_result) {
    std::cerr << "Buffer creation failed\n";
    return;
  }

  Buffer buffer = std::move(buffer_result.value());
  std::cout << "Created buffer: " << buffer.size() << " bytes on "
            << device_type_str(buffer.device()) << "\n";

  // Access as typed span (zero-copy view)
  auto span = buffer.as_span<f32>();
  std::cout << "Buffer contains " << span.size() << " floats\n";

  // Fill with pattern
  for (size_t i = 0; i < span.size(); ++i) {
    span[i] = static_cast<f32>(i);
  }

  // Read back
  std::cout << "First 10 elements: ";
  for (size_t i = 0; i < std::min(size_t(10), span.size()); ++i) {
    std::cout << span[i] << " ";
  }
  std::cout << "\n";

  // Clone the buffer
  auto clone_result = buffer.clone();
  if (clone_result) {
    std::cout << "Buffer cloned successfully\n";
    // clone_result.value() will be automatically freed
  }

  // buffer will be automatically freed when it goes out of scope
  std::cout << "Buffer will be freed automatically\n";
}

// ============================================================================
// Example 4: C++20 Concepts
// ============================================================================

// Generic function using Concepts
template <FloatingPoint T>
T compute_mean(std::span<const T> data) {
  if (data.empty()) {
    return T{0};
  }

  T sum = 0;
  for (T value : data) {
    sum += value;
  }
  return sum / static_cast<T>(data.size());
}

void example_concepts() {
  std::cout << "\n=== Example 4: C++20 Concepts ===\n";

  // Create data
  f32 data_f32[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  f64 data_f64[] = {1.0, 2.0, 3.0, 4.0, 5.0};

  // Use the generic function
  f32 mean_f32 = compute_mean<f32>(data_f32);
  f64 mean_f64 = compute_mean<f64>(data_f64);

  std::cout << "Mean (f32): " << mean_f32 << "\n";
  std::cout << "Mean (f64): " << mean_f64 << "\n";

  // This would cause a compile error:
  // int data_int[] = {1, 2, 3, 4, 5};
  // compute_mean<int>(data_int);  // Error: int doesn't satisfy FloatingPoint
}

// ============================================================================
// Example 5: Type System
// ============================================================================

void example_type_system() {
  std::cout << "\n=== Example 5: Type System ===\n";

  // DataType sizes
  std::cout << "DataType sizes:\n";
  std::cout << "  Float32: " << data_type_size(DataType::Float32) << " bytes\n";
  std::cout << "  Float64: " << data_type_size(DataType::Float64) << " bytes\n";
  std::cout << "  Int32: " << data_type_size(DataType::Int32) << " bytes\n";

  // DataType names
  std::cout << "\nDataType names:\n";
  std::cout << "  " << static_cast<int>(DataType::Float32) << " -> "
            << data_type_str(DataType::Float32) << "\n";
  std::cout << "  " << static_cast<int>(DataType::Int32) << " -> "
            << data_type_str(DataType::Int32) << "\n";

  // C++ type to DataType mapping
  std::cout << "\nC++ type to DataType:\n";
  std::cout << "  f32 -> " << data_type_str(cpp_type_to_data_type_v<f32>)
            << "\n";
  std::cout << "  i32 -> " << data_type_str(cpp_type_to_data_type_v<i32>)
            << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "PhotonInfer PR1 Examples\n";
  std::cout << "========================\n";

  try {
    example_error_handling();
    example_allocator();
    example_buffer();
    example_concepts();
    example_type_system();

    std::cout << "\n=== All examples completed successfully! ===\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}
