/**
 * @file test_add.cpp
 * @brief Unit tests for Add operator
 */

#include <gtest/gtest.h>
#include <photon/ops/add.hpp>
#include <photon/core/tensor.hpp>
#include <cmath>
#include <random>
#include <chrono>

using namespace photon;

// Test 1: Basic construction
TEST(AddOpTest, BasicConstruction) {
  AddOp op;

  EXPECT_FALSE(op.use_naive());
  EXPECT_EQ(op.device(), DeviceType::CPU);
}

// Test 2: Simple addition
TEST(AddOpTest, SimpleAddition) {
  const i32 size = 5;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // input1 = [1, 2, 3, 4, 5]
  f32* ptr1 = input1.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = static_cast<f32>(i + 1);
  }

  // input2 = [10, 20, 30, 40, 50]
  f32* ptr2 = input2.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr2[i] = static_cast<f32>((i + 1) * 10);
  }

  ASSERT_TRUE(op.forward(input1.value(), input2.value(), output.value()));

  // Verify: output = [11, 22, 33, 44, 55]
  f32* out_ptr = output.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    f32 expected = static_cast<f32>((i + 1) + (i + 1) * 10);
    EXPECT_FLOAT_EQ(out_ptr[i], expected) << "Mismatch at index " << i;
  }
}

// Test 3: Zero addition
TEST(AddOpTest, ZeroAddition) {
  const i32 size = 100;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // input1 = [1, 2, 3, ...]
  f32* ptr1 = input1.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = static_cast<f32>(i + 1);
  }

  // input2 = [0, 0, 0, ...]
  f32* ptr2 = input2.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr2[i] = 0.0f;
  }

  ASSERT_TRUE(op.forward(input1.value(), input2.value(), output.value()));

  // Verify: output = input1
  f32* out_ptr = output.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], ptr1[i]);
  }
}

// Test 4: Negative numbers
TEST(AddOpTest, NegativeNumbers) {
  const i32 size = 4;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  f32* ptr1 = input1.value().ptr<f32>();
  ptr1[0] = 5.0f;
  ptr1[1] = -3.0f;
  ptr1[2] = 2.5f;
  ptr1[3] = -7.2f;

  f32* ptr2 = input2.value().ptr<f32>();
  ptr2[0] = -5.0f;
  ptr2[1] = 3.0f;
  ptr2[2] = -1.5f;
  ptr2[3] = 7.2f;

  ASSERT_TRUE(op.forward(input1.value(), input2.value(), output.value()));

  f32* out_ptr = output.value().ptr<f32>();
  EXPECT_FLOAT_EQ(out_ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(out_ptr[1], 0.0f);
  EXPECT_FLOAT_EQ(out_ptr[2], 1.0f);
  EXPECT_FLOAT_EQ(out_ptr[3], 0.0f);
}

// Test 5: Large tensor
TEST(AddOpTest, LargeTensor) {
  const i32 size = 10000;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  std::mt19937 gen(42);
  std::uniform_real_distribution<f32> dis(-100.0f, 100.0f);

  f32* ptr1 = input1.value().ptr<f32>();
  f32* ptr2 = input2.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = dis(gen);
    ptr2[i] = dis(gen);
  }

  ASSERT_TRUE(op.forward(input1.value(), input2.value(), output.value()));

  // Verify a few random samples
  f32* out_ptr = output.value().ptr<f32>();
  for (i32 i = 0; i < 100; i += 10) {
    EXPECT_NEAR(out_ptr[i], ptr1[i] + ptr2[i], 1e-5f);
  }
}

// Test 6: Float64 support
TEST(AddOpTest, Float64Support) {
  const i32 size = 10;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float64, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float64, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float64, DeviceType::CPU);
  ASSERT_TRUE(output);

  f64* ptr1 = input1.value().ptr<f64>();
  f64* ptr2 = input2.value().ptr<f64>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = static_cast<f64>(i) * 0.1;
    ptr2[i] = static_cast<f64>(i) * 0.2;
  }

  ASSERT_TRUE(op.forward(input1.value(), input2.value(), output.value()));

  f64* out_ptr = output.value().ptr<f64>();
  for (i32 i = 0; i < size; ++i) {
    f64 expected = static_cast<f64>(i) * 0.3;
    EXPECT_NEAR(out_ptr[i], expected, 1e-10);
  }
}

// Test 7: Naive vs Eigen implementation
TEST(AddOpTest, NaiveVsEigen) {
  const i32 size = 1000;

  AddOp op_naive(/*use_naive=*/true);
  AddOp op_eigen(/*use_naive=*/false);
  ASSERT_TRUE(op_naive.init());
  ASSERT_TRUE(op_eigen.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output_naive = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_naive);
  auto output_eigen = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output_eigen);

  std::mt19937 gen(123);
  std::normal_distribution<f32> dis(0.0f, 10.0f);

  f32* ptr1 = input1.value().ptr<f32>();
  f32* ptr2 = input2.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = dis(gen);
    ptr2[i] = dis(gen);
  }

  ASSERT_TRUE(op_naive.forward(input1.value(), input2.value(), output_naive.value()));
  ASSERT_TRUE(op_eigen.forward(input1.value(), input2.value(), output_eigen.value()));

  // Compare outputs
  f32* out_naive_ptr = output_naive.value().ptr<f32>();
  f32* out_eigen_ptr = output_eigen.value().ptr<f32>();

  for (i32 i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(out_naive_ptr[i], out_eigen_ptr[i])
        << "Mismatch at index " << i;
  }
}

// Test 8: In-place addition (output aliases input1)
TEST(AddOpTest, InPlaceAddition) {
  const i32 size = 100;

  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);

  f32* ptr1 = input1.value().ptr<f32>();
  f32* ptr2 = input2.value().ptr<f32>();

  std::vector<f32> expected(size);
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = static_cast<f32>(i);
    ptr2[i] = static_cast<f32>(i * 2);
    expected[i] = static_cast<f32>(i + i * 2);
  }

  // In-place: input1 = input1 + input2
  ASSERT_TRUE(op.forward(input1.value(), input2.value(), input1.value()));

  for (i32 i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(ptr1[i], expected[i]);
  }
}

// Test 9: Error handling - size mismatch
TEST(AddOpTest, ErrorSizeMismatch) {
  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({50}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // Should fail due to size mismatch
  EXPECT_FALSE(op.forward(input1.value(), input2.value(), output.value()));
}

// Test 10: Error handling - dtype mismatch
TEST(AddOpTest, ErrorDtypeMismatch) {
  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({100}, DataType::Float64, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // Should fail due to dtype mismatch
  EXPECT_FALSE(op.forward(input1.value(), input2.value(), output.value()));
}

// Test 11: Error handling - empty tensors
TEST(AddOpTest, ErrorEmptyTensors) {
  AddOp op;
  ASSERT_TRUE(op.init());

  auto input1 = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({100}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);

  Tensor empty_tensor;

  // Empty input1
  EXPECT_FALSE(op.forward(empty_tensor, input2.value(), input1.value()));

  // Empty input2
  EXPECT_FALSE(op.forward(input1.value(), empty_tensor, input1.value()));

  // Empty output
  EXPECT_FALSE(op.forward(input1.value(), input2.value(), empty_tensor));
}

// Test 12: Benchmark
TEST(AddOpTest, Benchmark) {
  const i32 size = 1024;
  const i32 iterations = 10000;

  AddOp op_naive(/*use_naive=*/true);
  AddOp op_eigen(/*use_naive=*/false);
  ASSERT_TRUE(op_naive.init());
  ASSERT_TRUE(op_eigen.init());

  auto input1 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input1);
  auto input2 = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(input2);
  auto output = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
  ASSERT_TRUE(output);

  // Initialize with random data
  std::mt19937 gen(456);
  std::uniform_real_distribution<f32> dis(-1.0f, 1.0f);
  f32* ptr1 = input1.value().ptr<f32>();
  f32* ptr2 = input2.value().ptr<f32>();
  for (i32 i = 0; i < size; ++i) {
    ptr1[i] = dis(gen);
    ptr2[i] = dis(gen);
  }

  // Benchmark naive
  auto start_naive = std::chrono::high_resolution_clock::now();
  for (i32 iter = 0; iter < iterations; ++iter) {
    op_naive.forward(input1.value(), input2.value(), output.value());
  }
  auto end_naive = std::chrono::high_resolution_clock::now();
  auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(
      end_naive - start_naive).count();

  // Benchmark Eigen
  auto start_eigen = std::chrono::high_resolution_clock::now();
  for (i32 iter = 0; iter < iterations; ++iter) {
    op_eigen.forward(input1.value(), input2.value(), output.value());
  }
  auto end_eigen = std::chrono::high_resolution_clock::now();
  auto duration_eigen = std::chrono::duration_cast<std::chrono::microseconds>(
      end_eigen - start_eigen).count();

  f64 speedup = static_cast<f64>(duration_naive) / static_cast<f64>(duration_eigen);

  std::cout << "\nAdd Benchmark (size=" << size << ", " << iterations << " iterations):\n";
  std::cout << "  Naive:  " << duration_naive << " μs total, "
            << static_cast<f64>(duration_naive) / iterations << " μs/iter\n";
  std::cout << "  Eigen:  " << duration_eigen << " μs total, "
            << static_cast<f64>(duration_eigen) / iterations << " μs/iter\n";
  std::cout << "  Speedup: " << speedup << "x\n";

  // For simple element-wise operations, Eigen may not always be faster due to overhead
  // Just verify both implementations produce correct results (tested in NaiveVsEigen)
  EXPECT_GT(speedup, 0.5);  // Sanity check: Eigen shouldn't be much slower
}
