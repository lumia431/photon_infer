/**
 * @file types.hpp
 * @brief Fundamental type definitions and concepts for PhotonInfer
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 *
 * This file provides modern C++20 type definitions, concepts, and traits
 * that form the foundation of the PhotonInfer framework.
 */

#ifndef PHOTON_CORE_TYPES_HPP
#define PHOTON_CORE_TYPES_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace photon {

// ============================================================================
// Integer Types
// ============================================================================

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

// ============================================================================
// Floating Point Types
// ============================================================================

using f32 = float;
using f64 = double;

// ============================================================================
// Device Types
// ============================================================================

/**
 * @enum DeviceType
 * @brief Enumeration of supported compute devices
 */
enum class DeviceType : u8 {
  CPU = 0,   ///< CPU device
  CUDA = 1,  ///< NVIDIA CUDA GPU
};

/**
 * @brief Convert DeviceType to string representation
 */
constexpr const char* device_type_str(DeviceType type) noexcept {
  switch (type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    default:
      return "Unknown";
  }
}

// ============================================================================
// Data Types
// ============================================================================

/**
 * @enum DataType
 * @brief Enumeration of supported tensor data types
 */
enum class DataType : u8 {
  Float32 = 0,  ///< 32-bit floating point
  Float64 = 1,  ///< 64-bit floating point
  Int8 = 2,     ///< 8-bit signed integer
  Int16 = 3,    ///< 16-bit signed integer
  Int32 = 4,    ///< 32-bit signed integer
  Int64 = 5,    ///< 64-bit signed integer
  UInt8 = 6,    ///< 8-bit unsigned integer
  UInt16 = 7,   ///< 16-bit unsigned integer
  UInt32 = 8,   ///< 32-bit unsigned integer
  UInt64 = 9,   ///< 64-bit unsigned integer
};

/**
 * @brief Get the size in bytes of a DataType
 */
constexpr usize data_type_size(DataType type) noexcept {
  switch (type) {
    case DataType::Float32:
    case DataType::Int32:
    case DataType::UInt32:
      return 4;
    case DataType::Float64:
    case DataType::Int64:
    case DataType::UInt64:
      return 8;
    case DataType::Int8:
    case DataType::UInt8:
      return 1;
    case DataType::Int16:
    case DataType::UInt16:
      return 2;
    default:
      return 0;
  }
}

/**
 * @brief Convert DataType to string representation
 */
constexpr const char* data_type_str(DataType type) noexcept {
  switch (type) {
    case DataType::Float32:
      return "float32";
    case DataType::Float64:
      return "float64";
    case DataType::Int8:
      return "int8";
    case DataType::Int16:
      return "int16";
    case DataType::Int32:
      return "int32";
    case DataType::Int64:
      return "int64";
    case DataType::UInt8:
      return "uint8";
    case DataType::UInt16:
      return "uint16";
    case DataType::UInt32:
      return "uint32";
    case DataType::UInt64:
      return "uint64";
    default:
      return "unknown";
  }
}

// ============================================================================
// Type Traits and Concepts
// ============================================================================

/**
 * @brief Concept that constrains T to be a floating-point type
 */
template <typename T>
concept FloatingPoint = std::floating_point<T>;

/**
 * @brief Concept that constrains T to be an integral type
 */
template <typename T>
concept Integral = std::integral<T>;

/**
 * @brief Concept that constrains T to be a numeric type (integer or float)
 */
template <typename T>
concept Numeric = FloatingPoint<T> || Integral<T>;

/**
 * @brief Concept that constrains T to be a signed numeric type
 */
template <typename T>
concept SignedNumeric = Numeric<T> && std::is_signed_v<T>;

/**
 * @brief Concept that constrains T to be an unsigned numeric type
 */
template <typename T>
concept UnsignedNumeric = Integral<T> && std::is_unsigned_v<T>;

// ============================================================================
// DataType to C++ Type Mapping
// ============================================================================

/**
 * @brief Map DataType enum to actual C++ type
 */
template <DataType DT>
struct DataTypeMap;

template <>
struct DataTypeMap<DataType::Float32> {
  using type = f32;
};

template <>
struct DataTypeMap<DataType::Float64> {
  using type = f64;
};

template <>
struct DataTypeMap<DataType::Int8> {
  using type = i8;
};

template <>
struct DataTypeMap<DataType::Int16> {
  using type = i16;
};

template <>
struct DataTypeMap<DataType::Int32> {
  using type = i32;
};

template <>
struct DataTypeMap<DataType::Int64> {
  using type = i64;
};

template <>
struct DataTypeMap<DataType::UInt8> {
  using type = u8;
};

template <>
struct DataTypeMap<DataType::UInt16> {
  using type = u16;
};

template <>
struct DataTypeMap<DataType::UInt32> {
  using type = u32;
};

template <>
struct DataTypeMap<DataType::UInt64> {
  using type = u64;
};

/// Helper alias for DataTypeMap
template <DataType DT>
using data_type_t = typename DataTypeMap<DT>::type;

// ============================================================================
// C++ Type to DataType Mapping
// ============================================================================

/**
 * @brief Map C++ type to DataType enum
 */
template <typename T>
struct CppTypeToDataType;

template <>
struct CppTypeToDataType<f32> {
  static constexpr DataType value = DataType::Float32;
};

template <>
struct CppTypeToDataType<f64> {
  static constexpr DataType value = DataType::Float64;
};

template <>
struct CppTypeToDataType<i8> {
  static constexpr DataType value = DataType::Int8;
};

template <>
struct CppTypeToDataType<i16> {
  static constexpr DataType value = DataType::Int16;
};

template <>
struct CppTypeToDataType<i32> {
  static constexpr DataType value = DataType::Int32;
};

template <>
struct CppTypeToDataType<i64> {
  static constexpr DataType value = DataType::Int64;
};

template <>
struct CppTypeToDataType<u8> {
  static constexpr DataType value = DataType::UInt8;
};

template <>
struct CppTypeToDataType<u16> {
  static constexpr DataType value = DataType::UInt16;
};

template <>
struct CppTypeToDataType<u32> {
  static constexpr DataType value = DataType::UInt32;
};

template <>
struct CppTypeToDataType<u64> {
  static constexpr DataType value = DataType::UInt64;
};

/// Helper variable template for type to DataType
template <typename T>
inline constexpr DataType cpp_type_to_data_type_v = CppTypeToDataType<T>::value;

}  // namespace photon

#endif  // PHOTON_CORE_TYPES_HPP
