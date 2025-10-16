/**
 * @file test_tensor.cpp
 * @brief Unit tests for Tensor class
 */

#include "photon/core/tensor.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

using namespace photon;

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST(TensorTest, Create) {
  auto result = Tensor::create({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  EXPECT_EQ(tensor.dims(), std::vector<int32_t>({2, 3}));
  EXPECT_EQ(tensor.dtype(), DataType::Float32);
  EXPECT_EQ(tensor.device(), DeviceType::CPU);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_FALSE(tensor.empty());
}

TEST(TensorTest, Zeros) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  const f32* data = tensor.ptr<f32>();

  for (usize i = 0; i < tensor.size(); ++i) {
    EXPECT_EQ(data[i], 0.0f);
  }
}

TEST(TensorTest, FromVector) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto result = Tensor::from_vector(data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.dim(0), 6);

  const f32* ptr = tensor.ptr<f32>();
  for (usize i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(TensorTest, FromMatrix) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto result = Tensor::from_matrix(2, 3, data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.dim(0), 2);
  EXPECT_EQ(tensor.dim(1), 3);
  EXPECT_EQ(tensor.size(), 6);

  const f32* ptr = tensor.ptr<f32>();
  for (usize i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(TensorTest, FromMatrixSizeMismatch) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f};
  auto result = Tensor::from_matrix(2, 3, data);
  EXPECT_FALSE(result.is_ok());
  EXPECT_EQ(result.error().code(), ErrorCode::InvalidArgument);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(TensorTest, MoveConstruction) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor1 = std::move(result.value());
  Tensor tensor2 = std::move(tensor1);

  EXPECT_EQ(tensor2.size(), 6);
  EXPECT_EQ(tensor2.ndim(), 2);
}

TEST(TensorTest, MoveAssignment) {
  auto result1 = Tensor::zeros({2, 3}, DataType::Float32);
  auto result2 = Tensor::zeros({3, 4}, DataType::Float32);
  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  Tensor tensor1 = std::move(result1.value());
  Tensor tensor2 = std::move(result2.value());

  tensor2 = std::move(tensor1);
  EXPECT_EQ(tensor2.size(), 6);
  EXPECT_EQ(tensor2.ndim(), 2);
}

// ============================================================================
// Data Access Tests
// ============================================================================

TEST(TensorTest, DataPointerAccess) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());

  // Mutable access
  f32* data_ptr = tensor.ptr<f32>();
  EXPECT_NE(data_ptr, nullptr);
  data_ptr[0] = 42.0f;

  // Const access
  const Tensor& const_tensor = tensor;
  const f32* const_ptr = const_tensor.ptr<f32>();
  EXPECT_EQ(const_ptr[0], 42.0f);
}

TEST(TensorTest, DataPointerWithOffset) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  f32* ptr_offset = tensor.ptr<f32>(3);
  ptr_offset[0] = 99.0f;

  f32* ptr_base = tensor.ptr<f32>();
  EXPECT_EQ(ptr_base[3], 99.0f);
}

TEST(TensorTest, IndexAccess) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = Tensor::from_vector(data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());

  EXPECT_EQ(tensor.index<f32>(0), 1.0f);
  EXPECT_EQ(tensor.index<f32>(1), 2.0f);
  EXPECT_EQ(tensor.index<f32>(2), 3.0f);
  EXPECT_EQ(tensor.index<f32>(3), 4.0f);

  tensor.index<f32>(2) = 10.0f;
  EXPECT_EQ(tensor.index<f32>(2), 10.0f);
}

// ============================================================================
// Eigen Integration Tests
// ============================================================================

TEST(TensorTest, VectorMap) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = Tensor::from_vector(data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());

  // Get Eigen map
  auto vec_map = tensor.vector_map<f32>();
  EXPECT_EQ(vec_map.size(), 4);

  // Use Eigen operations
  vec_map = vec_map * 2.0f;

  // Verify results
  EXPECT_EQ(tensor.ptr<f32>()[0], 2.0f);
  EXPECT_EQ(tensor.ptr<f32>()[1], 4.0f);
  EXPECT_EQ(tensor.ptr<f32>()[2], 6.0f);
  EXPECT_EQ(tensor.ptr<f32>()[3], 8.0f);
}

TEST(TensorTest, MatrixMap) {
  // Create 2x3 matrix
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto result = Tensor::from_matrix(2, 3, data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());

  // Get Eigen map (row-major)
  auto mat_map = tensor.matrix_map<f32>();
  EXPECT_EQ(mat_map.rows(), 2);
  EXPECT_EQ(mat_map.cols(), 3);

  // Access elements
  EXPECT_EQ(mat_map(0, 0), 1.0f);
  EXPECT_EQ(mat_map(0, 1), 2.0f);
  EXPECT_EQ(mat_map(0, 2), 3.0f);
  EXPECT_EQ(mat_map(1, 0), 4.0f);
  EXPECT_EQ(mat_map(1, 1), 5.0f);
  EXPECT_EQ(mat_map(1, 2), 6.0f);
}

TEST(TensorTest, MatrixMapScaling) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = Tensor::from_matrix(2, 2, data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  auto mat_map = tensor.matrix_map<f32>();

  // Scale matrix
  mat_map = mat_map * 3.0f;

  EXPECT_EQ(mat_map(0, 0), 3.0f);  // 1*3
  EXPECT_EQ(mat_map(0, 1), 6.0f);  // 2*3
  EXPECT_EQ(mat_map(1, 0), 9.0f);  // 3*3
  EXPECT_EQ(mat_map(1, 1), 12.0f); // 4*3
}

TEST(TensorTest, MatrixMultiplication) {
  // Matrix A: 2x3
  std::vector<f32> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto result_a = Tensor::from_matrix(2, 3, data_a);
  ASSERT_TRUE(result_a.is_ok());
  Tensor tensor_a = std::move(result_a.value());

  // Matrix B: 3x2
  std::vector<f32> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  auto result_b = Tensor::from_matrix(3, 2, data_b);
  ASSERT_TRUE(result_b.is_ok());
  Tensor tensor_b = std::move(result_b.value());

  // Result C: 2x2
  auto result_c = Tensor::zeros({2, 2}, DataType::Float32);
  ASSERT_TRUE(result_c.is_ok());
  Tensor tensor_c = std::move(result_c.value());

  // Matrix multiplication using Eigen
  const Tensor& const_a = tensor_a;
  const Tensor& const_b = tensor_b;
  auto map_a = const_a.matrix_map<f32>();
  auto map_b = const_b.matrix_map<f32>();
  auto map_c = tensor_c.matrix_map<f32>();

  map_c = map_a * map_b;

  // Expected result:
  // C[0,0] = 1*7 + 2*9 + 3*11 = 58
  // C[0,1] = 1*8 + 2*10 + 3*12 = 64
  // C[1,0] = 4*7 + 5*9 + 6*11 = 139
  // C[1,1] = 4*8 + 5*10 + 6*12 = 154

  EXPECT_FLOAT_EQ(map_c(0, 0), 58.0f);
  EXPECT_FLOAT_EQ(map_c(0, 1), 64.0f);
  EXPECT_FLOAT_EQ(map_c(1, 0), 139.0f);
  EXPECT_FLOAT_EQ(map_c(1, 1), 154.0f);
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST(TensorTest, Reshape) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());

  // Valid reshape
  auto reshape_result = tensor.reshape({3, 2});
  EXPECT_TRUE(reshape_result.is_ok());
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.dim(0), 3);
  EXPECT_EQ(tensor.dim(1), 2);

  // Invalid reshape (element count mismatch)
  auto invalid_result = tensor.reshape({2, 2});
  EXPECT_FALSE(invalid_result.is_ok());
  EXPECT_EQ(invalid_result.error().code(), ErrorCode::InvalidArgument);
}

TEST(TensorTest, ReshapeToVector) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  auto reshape_result = tensor.reshape({6});
  EXPECT_TRUE(reshape_result.is_ok());
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.dim(0), 6);
  EXPECT_EQ(tensor.size(), 6);
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST(TensorTest, Clone) {
  std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto result = Tensor::from_matrix(2, 2, data);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor1 = std::move(result.value());
  auto clone_result = tensor1.clone();
  ASSERT_TRUE(clone_result.is_ok());

  Tensor tensor2 = std::move(clone_result.value());

  // Verify clone has same properties
  EXPECT_EQ(tensor2.dims(), tensor1.dims());
  EXPECT_EQ(tensor2.dtype(), tensor1.dtype());
  EXPECT_EQ(tensor2.device(), tensor1.device());

  // Verify data is copied
  const f32* ptr1 = tensor1.ptr<f32>();
  const f32* ptr2 = tensor2.ptr<f32>();
  for (usize i = 0; i < tensor1.size(); ++i) {
    EXPECT_EQ(ptr1[i], ptr2[i]);
  }

  // Verify independent storage
  tensor2.ptr<f32>()[0] = 999.0f;
  EXPECT_NE(ptr1[0], tensor2.ptr<f32>()[0]);
}

// ============================================================================
// String Representation Tests
// ============================================================================

TEST(TensorTest, ToString) {
  auto result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  std::string str = tensor.to_string();

  EXPECT_NE(str.find("shape=[2, 3]"), std::string::npos);
  EXPECT_NE(str.find("dtype=float32"), std::string::npos);
  EXPECT_NE(str.find("size=6"), std::string::npos);
}

// ============================================================================
// Different Data Types Tests
// ============================================================================

TEST(TensorTest, DifferentDataTypes) {
  // Float32
  auto f32_result = Tensor::zeros({2, 3}, DataType::Float32);
  ASSERT_TRUE(f32_result.is_ok());
  EXPECT_EQ(f32_result.value().dtype(), DataType::Float32);

  // Int32
  auto i32_result = Tensor::zeros({2, 3}, DataType::Int32);
  ASSERT_TRUE(i32_result.is_ok());
  EXPECT_EQ(i32_result.value().dtype(), DataType::Int32);

  // Float64
  auto f64_result = Tensor::zeros({2, 3}, DataType::Float64);
  ASSERT_TRUE(f64_result.is_ok());
  EXPECT_EQ(f64_result.value().dtype(), DataType::Float64);
}

// ============================================================================
// Large Tensor Test
// ============================================================================

TEST(TensorTest, LargeTensor) {
  // Create a large tensor (10MB)
  constexpr usize size = 1024 * 1024 * 10 / sizeof(f32);  // ~10MB
  auto result = Tensor::zeros({static_cast<int32_t>(size)}, DataType::Float32);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  EXPECT_EQ(tensor.size(), size);
}

// ============================================================================
// CUDA Tests
// ============================================================================

#ifdef PHOTON_USE_CUDA
TEST(TensorTest, CUDACreation) {
  auto result =
      Tensor::create({2, 3}, DataType::Float32, DeviceType::CUDA);
  ASSERT_TRUE(result.is_ok());

  Tensor tensor = std::move(result.value());
  EXPECT_EQ(tensor.device(), DeviceType::CUDA);
  EXPECT_EQ(tensor.size(), 6);
}
#endif
