/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once

/**
 * @file error.hpp
 * @brief Error handling system using Result<T, E> pattern
 * @author PhotonInfer Team
 * @version 0.1.0
 * @date 2025-01-15
 *
 * This file provides a modern error handling system inspired by Rust's Result<T, E>
 * and C++23's std::expected. It encourages explicit error handling and provides
 * a type-safe way to propagate errors through the call stack.
 */


#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "types.hpp"

namespace photon {

// ============================================================================
// Error Codes
// ============================================================================

/**
 * @enum ErrorCode
 * @brief Standard error codes used throughout PhotonInfer
 */
enum class ErrorCode : u32 {
  // Success
  Ok = 0,

  // Memory errors (1-99)
  OutOfMemory = 1,
  InvalidAlignment = 2,
  NullPointer = 3,
  BufferOverflow = 4,

  // I/O errors (100-199)
  FileNotFound = 100,
  FileReadError = 101,
  FileWriteError = 102,
  InvalidFileFormat = 103,

  // Tensor errors (200-299)
  InvalidShape = 200,
  InvalidDtype = 201,
  ShapeMismatch = 202,
  DeviceMismatch = 203,
  InvalidIndex = 204,

  // Model errors (300-399)
  ModelLoadError = 300,
  InvalidModelFormat = 301,
  UnsupportedModel = 302,
  WeightMismatch = 303,
  ModelParseError = 304,
  IOError = 305,

  // Operator errors (400-499)
  InvalidOperator = 400,
  OperatorNotImplemented = 401,
  InvalidOperand = 402,

  // CUDA errors (500-599)
  CudaError = 500,
  CudaOutOfMemory = 501,
  CudaLaunchFailed = 502,
  CudaInvalidDevice = 503,

  // Generic errors (900-999)
  InvalidArgument = 900,
  NotImplemented = 901,
  InternalError = 902,
  Unknown = 999,
};

/**
 * @brief Convert ErrorCode to string representation
 */
constexpr std::string_view error_code_str(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Ok:
      return "Ok";

    // Memory
    case ErrorCode::OutOfMemory:
      return "OutOfMemory";
    case ErrorCode::InvalidAlignment:
      return "InvalidAlignment";
    case ErrorCode::NullPointer:
      return "NullPointer";
    case ErrorCode::BufferOverflow:
      return "BufferOverflow";

    // I/O
    case ErrorCode::FileNotFound:
      return "FileNotFound";
    case ErrorCode::FileReadError:
      return "FileReadError";
    case ErrorCode::FileWriteError:
      return "FileWriteError";
    case ErrorCode::InvalidFileFormat:
      return "InvalidFileFormat";

    // Tensor
    case ErrorCode::InvalidShape:
      return "InvalidShape";
    case ErrorCode::InvalidDtype:
      return "InvalidDtype";
    case ErrorCode::ShapeMismatch:
      return "ShapeMismatch";
    case ErrorCode::DeviceMismatch:
      return "DeviceMismatch";
    case ErrorCode::InvalidIndex:
      return "InvalidIndex";

    // Model
    case ErrorCode::ModelLoadError:
      return "ModelLoadError";
    case ErrorCode::InvalidModelFormat:
      return "InvalidModelFormat";
    case ErrorCode::UnsupportedModel:
      return "UnsupportedModel";
    case ErrorCode::WeightMismatch:
      return "WeightMismatch";
    case ErrorCode::ModelParseError:
      return "ModelParseError";
    case ErrorCode::IOError:
      return "IOError";

    // Operator
    case ErrorCode::InvalidOperator:
      return "InvalidOperator";
    case ErrorCode::OperatorNotImplemented:
      return "OperatorNotImplemented";
    case ErrorCode::InvalidOperand:
      return "InvalidOperand";

    // CUDA
    case ErrorCode::CudaError:
      return "CudaError";
    case ErrorCode::CudaOutOfMemory:
      return "CudaOutOfMemory";
    case ErrorCode::CudaLaunchFailed:
      return "CudaLaunchFailed";
    case ErrorCode::CudaInvalidDevice:
      return "CudaInvalidDevice";

    // Generic
    case ErrorCode::InvalidArgument:
      return "InvalidArgument";
    case ErrorCode::NotImplemented:
      return "NotImplemented";
    case ErrorCode::InternalError:
      return "InternalError";
    case ErrorCode::Unknown:
      return "Unknown";

    default:
      return "UnknownErrorCode";
  }
}

// ============================================================================
// Error Class
// ============================================================================

/**
 * @class Error
 * @brief Represents an error with code and optional message
 */
class Error {
 public:
  /**
   * @brief Default constructor creates an Ok error
   */
  constexpr Error() noexcept : code_(ErrorCode::Ok), message_() {}

  /**
   * @brief Construct error with code only
   */
  constexpr explicit Error(ErrorCode code) noexcept : code_(code), message_() {}

  /**
   * @brief Construct error with code and message
   */
  Error(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  /**
   * @brief Get the error code
   */
  [[nodiscard]] constexpr ErrorCode code() const noexcept { return code_; }

  /**
   * @brief Get the error message
   */
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  /**
   * @brief Check if this is an Ok (no error)
   */
  [[nodiscard]] constexpr bool is_ok() const noexcept {
    return code_ == ErrorCode::Ok;
  }

  /**
   * @brief Check if this is an error (not Ok)
   */
  [[nodiscard]] constexpr bool is_err() const noexcept {
    return code_ != ErrorCode::Ok;
  }

  /**
   * @brief Get full error description
   */
  [[nodiscard]] std::string to_string() const {
    if (is_ok()) {
      return "Ok";
    }
    if (message_.empty()) {
      return std::string(error_code_str(code_));
    }
    return std::string(error_code_str(code_)) + ": " + message_;
  }

  /**
   * @brief Boolean conversion for easy checking
   */
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return is_ok();
  }

 private:
  ErrorCode code_;
  std::string message_;
};

// ============================================================================
// Result<T, E> Type
// ============================================================================

/**
 * @class Result
 * @brief A Result type similar to Rust's Result<T, E> or C++23's std::expected
 *
 * Result<T, E> represents either success (containing a value of type T) or
 * failure (containing an error of type E). This forces explicit error handling
 * and makes error propagation visible in the type system.
 *
 * @tparam T The type of the success value
 * @tparam E The type of the error (defaults to Error)
 */
template <typename T, typename E = Error>
class [[nodiscard]] Result {
 public:
  using value_type = T;
  using error_type = E;

  /**
   * @brief Construct a successful Result with a value
   */
  constexpr Result(T value) : storage_(std::move(value)) {}

  /**
   * @brief Construct a failed Result with an error
   */
  constexpr Result(E error) : storage_(std::move(error)) {}

  /**
   * @brief Check if Result contains a value (is Ok)
   */
  [[nodiscard]] constexpr bool is_ok() const noexcept {
    return std::holds_alternative<T>(storage_);
  }

  /**
   * @brief Check if Result contains an error (is Err)
   */
  [[nodiscard]] constexpr bool is_err() const noexcept {
    return std::holds_alternative<E>(storage_);
  }

  /**
   * @brief Boolean conversion (true if Ok)
   */
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return is_ok();
  }

  /**
   * @brief Get the value (must be Ok, otherwise undefined behavior)
   */
  [[nodiscard]] constexpr T& value() & { return std::get<T>(storage_); }

  [[nodiscard]] constexpr const T& value() const& {
    return std::get<T>(storage_);
  }

  [[nodiscard]] constexpr T&& value() && {
    return std::get<T>(std::move(storage_));
  }

  [[nodiscard]] constexpr const T&& value() const&& {
    return std::get<T>(std::move(storage_));
  }

  /**
   * @brief Get the error (must be Err, otherwise undefined behavior)
   */
  [[nodiscard]] constexpr E& error() & { return std::get<E>(storage_); }

  [[nodiscard]] constexpr const E& error() const& {
    return std::get<E>(storage_);
  }

  [[nodiscard]] constexpr E&& error() && {
    return std::get<E>(std::move(storage_));
  }

  [[nodiscard]] constexpr const E&& error() const&& {
    return std::get<E>(std::move(storage_));
  }

  /**
   * @brief Get value or return default if error
   */
  template <typename U>
  [[nodiscard]] constexpr T value_or(U&& default_value) const& {
    return is_ok() ? value() : static_cast<T>(std::forward<U>(default_value));
  }

  template <typename U>
  [[nodiscard]] constexpr T value_or(U&& default_value) && {
    return is_ok() ? std::move(value())
                   : static_cast<T>(std::forward<U>(default_value));
  }

  /**
   * @brief Unwrap value (panics if error)
   */
  [[nodiscard]] constexpr T unwrap() && {
    if (is_err()) {
      // In production code, this should use a proper assertion/panic mechanism
      std::terminate();
    }
    return std::move(value());
  }

  /**
   * @brief Unwrap error (panics if ok)
   */
  [[nodiscard]] constexpr E unwrap_err() && {
    if (is_ok()) {
      std::terminate();
    }
    return std::move(error());
  }

 private:
  std::variant<T, E> storage_;
};

/**
 * @brief Specialization for void success type
 */
template <typename E>
class [[nodiscard]] Result<void, E> {
 public:
  using value_type = void;
  using error_type = E;

  /**
   * @brief Construct a successful Result (void)
   */
  constexpr Result() : has_value_(true), error_() {}

  /**
   * @brief Construct a failed Result with an error
   */
  constexpr Result(E error) : has_value_(false), error_(std::move(error)) {}

  [[nodiscard]] constexpr bool is_ok() const noexcept { return has_value_; }

  [[nodiscard]] constexpr bool is_err() const noexcept { return !has_value_; }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return is_ok();
  }

  [[nodiscard]] constexpr E& error() & { return error_; }

  [[nodiscard]] constexpr const E& error() const& { return error_; }

  [[nodiscard]] constexpr E&& error() && { return std::move(error_); }

  constexpr void unwrap() const {
    if (is_err()) {
      std::terminate();
    }
  }

 private:
  bool has_value_;
  E error_;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Create a successful Result
 */
template <typename T>
[[nodiscard]] constexpr Result<T> Ok(T value) {
  return Result<T>(std::move(value));
}

/**
 * @brief Create a successful Result for void
 */
[[nodiscard]] constexpr Result<void> Ok() { return Result<void>(); }

/**
 * @brief Create a failed Result with Error
 */
template <typename T>
[[nodiscard]] constexpr Result<T> Err(Error error) {
  return Result<T>(std::move(error));
}

/**
 * @brief Create a failed Result with ErrorCode
 */
template <typename T>
[[nodiscard]] constexpr Result<T> Err(ErrorCode code) {
  return Result<T>(Error(code));
}

/**
 * @brief Create a failed Result with ErrorCode and message
 */
template <typename T>
[[nodiscard]] Result<T> Err(ErrorCode code, std::string message) {
  return Result<T>(Error(code, std::move(message)));
}

}  // namespace photon

