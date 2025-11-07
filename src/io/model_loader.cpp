/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#include "photon/io/model_loader.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>

namespace photon::model {

Result<ModelConfig> ModelLoader::read_config(std::FILE* file) {
  ModelConfig config;

  if (std::fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return Err<ModelConfig>(ErrorCode::ModelParseError,
                            "Failed to read ModelConfig from file");
  }

  // Validate configuration
  if (config.dim <= 0 || config.layer_num <= 0 || config.head_num <= 0) {
    return Err<ModelConfig>(ErrorCode::ModelParseError,
                            "Invalid model configuration: negative or zero dimensions");
  }

  return Ok(config);
}

Result<void> ModelLoader::mmap_file(const std::filesystem::path& path,
                                    i32& fd,
                                    void*& data,
                                    usize& file_size) {
  // Open file
  fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    return Err<void>(ErrorCode::FileNotFound,
                     "Failed to open model file: " + path.string());
  }

  // Get file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return Err<void>(ErrorCode::IOError, "Failed to get file size");
  }
  file_size = static_cast<usize>(sb.st_size);

  // Memory map the file
  data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED || data == nullptr) {
    close(fd);
    return Err<void>(ErrorCode::IOError, "Failed to mmap model file");
  }

  return Ok();
}

Result<ModelLoader::LoadedModel>
ModelLoader::load(const std::filesystem::path& model_path, bool is_quantized) {
  // Check file exists
  if (!std::filesystem::exists(model_path)) {
    return Err<LoadedModel>(ErrorCode::FileNotFound,
                            "Model file not found: " + model_path.string());
  }

  // Open file for reading config
  std::FILE* file = std::fopen(model_path.c_str(), "rb");
  if (!file) {
    return Err<LoadedModel>(ErrorCode::FileNotFound,
                            "Failed to open model file for reading");
  }

  // Read configuration
  auto config_result = read_config(file);
  if (!config_result) {
    std::fclose(file);
    return Err<LoadedModel>(config_result.error());
  }
  ModelConfig config = config_result.value();

  // Read group size for quantized models
  i32 group_size = 1;
  if (is_quantized) {
    if (std::fread(&group_size, sizeof(i32), 1, file) != 1) {
      std::fclose(file);
      return Err<LoadedModel>(ErrorCode::ModelParseError,
                              "Failed to read group_size for quantized model");
    }
  }

  std::fclose(file);

  // Create RawModelData based on quantization
  std::unique_ptr<RawModelData> raw_data;
  if (is_quantized) {
    raw_data = std::make_unique<RawModelDataInt8>();
  } else {
    raw_data = std::make_unique<RawModelDataFp32>();
  }

  // Memory map the file
  i32 fd;
  void* data;
  usize file_size;
  auto mmap_result = mmap_file(model_path, fd, data, file_size);
  if (!mmap_result) {
    return Err<LoadedModel>(mmap_result.error());
  }

  // Set up RawModelData
  raw_data->fd_ = fd;
  raw_data->data_ = data;
  raw_data->file_size_ = file_size;

  // Calculate weight data start position
  const usize offset = weight_offset(is_quantized);
  raw_data->weight_data_ = static_cast<i8*>(data) + offset;

  // Note: We'll compute transformer_config later when we have tokenizer vocab size
  // For now, use a placeholder
  TransformerConfig transformer_config = TransformerConfig::from_model_config(config, 0);

  return Ok(LoadedModel{.config = config,
                        .transformer_config = transformer_config,
                        .raw_data = std::move(raw_data),
                        .group_size = group_size,
                        .is_quantized = is_quantized});
}

template <typename WeightType>
  requires(std::same_as<WeightType, f32> || std::same_as<WeightType, i8>)
Result<ModelLoader::LoadedModelTyped<WeightType>>
ModelLoader::load_typed(const std::filesystem::path& model_path) {
  constexpr bool is_quantized = std::same_as<WeightType, i8>;

  // Use the non-typed load function
  auto result = load(model_path, is_quantized);
  if (!result) {
    return Err<LoadedModelTyped<WeightType>>(result.error());
  }

  // Cast the raw_data to the appropriate type
  auto loaded = std::move(result.value());
  auto typed_data = std::unique_ptr<typename RawModelDataFactory<WeightType>::type>(
      static_cast<typename RawModelDataFactory<WeightType>::type*>(
          loaded.raw_data.release()));

  return Ok(LoadedModelTyped<WeightType>{.config = loaded.config,
                                         .transformer_config = loaded.transformer_config,
                                         .raw_data = std::move(typed_data),
                                         .group_size = loaded.group_size});
}

// Explicit template instantiations
template Result<ModelLoader::LoadedModelTyped<f32>>
ModelLoader::load_typed<f32>(const std::filesystem::path&);

template Result<ModelLoader::LoadedModelTyped<i8>>
ModelLoader::load_typed<i8>(const std::filesystem::path&);

}  // namespace photon::model
