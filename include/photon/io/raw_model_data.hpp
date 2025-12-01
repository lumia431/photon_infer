/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once


#include "photon/core/types.hpp"
#include <memory>

namespace photon::model {

/**
 * @brief Base class for memory-mapped model data using RAII
 *
 * Uses mmap to efficiently load large model files without copying data.
 * Implements the Template Method pattern - subclasses define weight access.
 */
class RawModelData {
public:
  virtual ~RawModelData();

  // Non-copyable, movable
  RawModelData(const RawModelData&) = delete;
  RawModelData& operator=(const RawModelData&) = delete;
  RawModelData(RawModelData&&) noexcept = default;
  RawModelData& operator=(RawModelData&&) noexcept = default;

  /**
   * @brief Get pointer to weight data at given offset
   * @param offset Offset in elements (not bytes)
   * @return Pointer to weight data
   */
  [[nodiscard]] virtual const void* weight(usize offset) const = 0;

  /**
   * @brief Get raw data pointer
   */
  [[nodiscard]] const void* data() const noexcept { return data_; }

  /**
   * @brief Get file size in bytes
   */
  [[nodiscard]] usize file_size() const noexcept { return file_size_; }

  /**
   * @brief Check if data is valid
   */
  [[nodiscard]] bool is_valid() const noexcept {
    return data_ != nullptr && fd_ != -1;
  }

protected:
  RawModelData() = default;

  i32 fd_ = -1;             ///< File descriptor
  usize file_size_ = 0;     ///< Total file size
  void* data_ = nullptr;    ///< mmap'd memory region
  void* weight_data_ = nullptr; ///< Pointer to start of weight data

  friend class ModelLoader;
};

/**
 * @brief Float32 model data accessor
 */
class RawModelDataFp32 final : public RawModelData {
public:
  [[nodiscard]] const void* weight(usize offset) const override {
    return static_cast<const f32*>(weight_data_) + offset;
  }
};

/**
 * @brief Int8 quantized model data accessor
 */
class RawModelDataInt8 final : public RawModelData {
public:
  [[nodiscard]] const void* weight(usize offset) const override {
    return static_cast<const i8*>(weight_data_) + offset;
  }
};

/**
 * @brief Factory for creating appropriate RawModelData subclass
 */
template <typename T>
struct RawModelDataFactory;

template <>
struct RawModelDataFactory<f32> {
  using type = RawModelDataFp32;
};

template <>
struct RawModelDataFactory<i8> {
  using type = RawModelDataInt8;
};

} // namespace photon::model

