/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once


#include "photon/core/error.hpp"
#include "photon/arch/config.hpp"
#include "photon/io/raw_model_data.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace photon::model {

// Forward declarations
class ModelLoader;

/**
 * @brief Model file loader using mmap for efficient weight loading
 *
 * Loads Llama2.c format model files:
 * - Header: ModelConfig (7 * int32)
 * - [Optional] Group size (int32) for quantized models
 * - Weights: float32 or int8 data
 *
 * Uses C++20 features:
 * - Result<T> for error handling
 * - std::filesystem for path operations
 * - Template metaprogramming for type-safe weight access
 */
class ModelLoader {
public:
  /**
   * @brief Result of model loading operation
   */
  struct LoadedModel {
    ModelConfig config;                           ///< Raw model configuration
    TransformerConfig transformer_config;         ///< Computed transformer config
    std::unique_ptr<RawModelData> raw_data;      ///< Memory-mapped weight data
    i32 group_size = 1;                          ///< Quantization group size
    bool is_quantized = false;                   ///< Whether model is quantized
  };

  /**
   * @brief Typed model loading result (compile-time weight type)
   */
  template <typename WeightType>
  struct LoadedModelTyped {
    ModelConfig config;
    TransformerConfig transformer_config;
    std::unique_ptr<typename RawModelDataFactory<WeightType>::type> raw_data;
    i32 group_size = 1;
  };

  /**
   * @brief Load a model file with automatic precision detection
   * @param model_path Path to the model binary file
   * @param is_quantized Whether the model uses int8 quantization
   * @return Result containing loaded model data or error
   */
  [[nodiscard]] static Result<LoadedModel> load(
      const std::filesystem::path& model_path,
      bool is_quantized = false);

  /**
   * @brief Load model with explicit weight type (compile-time checked)
   * @tparam WeightType float or i8
   */
  template <typename WeightType>
    requires(std::same_as<WeightType, f32> || std::same_as<WeightType, i8>)
  [[nodiscard]] static Result<LoadedModelTyped<WeightType>> load_typed(
      const std::filesystem::path& model_path);

private:
  /**
   * @brief Read model configuration from file
   */
  static Result<ModelConfig> read_config(std::FILE* file);

  /**
   * @brief Memory-map the model file
   */
  static Result<void> mmap_file(
      const std::filesystem::path& path,
      i32& fd,
      void*& data,
      usize& file_size);

  /**
   * @brief Calculate weight data offset in file
   */
  static constexpr usize weight_offset(bool is_quantized) noexcept {
    usize offset = sizeof(ModelConfig);
    if (is_quantized) {
      offset += sizeof(i32); // group_size
    }
    return offset;
  }
};

} // namespace photon::model

