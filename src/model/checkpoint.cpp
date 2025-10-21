/**
 * @file checkpoint.cpp
 * @brief Checkpoint loader implementation
 * @version 0.1.0
 */

#include "photon/model/checkpoint.hpp"
#include "photon/model/llama_model.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

namespace photon::model {

Result<std::unique_ptr<CheckpointLoader>> CheckpointLoader::open(const std::string& checkpoint_path) {
  auto loader = std::unique_ptr<CheckpointLoader>(new CheckpointLoader());

  // Open file
  loader->fd_ = ::open(checkpoint_path.c_str(), O_RDONLY);
  if (loader->fd_ < 0) {
    return Err<std::unique_ptr<CheckpointLoader>>(
        ErrorCode::FileNotFound,
        "Failed to open checkpoint file: " + checkpoint_path);
  }

  // Get file size
  struct stat st;
  if (fstat(loader->fd_, &st) != 0) {
    ::close(loader->fd_);
    return Err<std::unique_ptr<CheckpointLoader>>(
        ErrorCode::FileReadError,
        "Failed to get file size: " + checkpoint_path);
  }
  loader->file_size_ = static_cast<usize>(st.st_size);

  // Memory-map the file
  loader->mmap_data_ = mmap(nullptr, loader->file_size_, PROT_READ, MAP_PRIVATE, loader->fd_, 0);
  if (loader->mmap_data_ == MAP_FAILED || loader->mmap_data_ == nullptr) {
    ::close(loader->fd_);
    return Err<std::unique_ptr<CheckpointLoader>>(
        ErrorCode::FileReadError,
        "Failed to mmap checkpoint file: " + checkpoint_path);
  }

  // Read header
  auto header_result = loader->read_header();
  if (!header_result) {
    munmap(loader->mmap_data_, loader->file_size_);
    ::close(loader->fd_);
    return Err<std::unique_ptr<CheckpointLoader>>(header_result.error());
  }

  // Set weight data pointer (after header)
  loader->weight_data_ = reinterpret_cast<const f32*>(
      static_cast<const u8*>(loader->mmap_data_) + sizeof(CheckpointHeader));

  return Ok(std::move(loader));
}

CheckpointLoader::~CheckpointLoader() {
  if (mmap_data_ != nullptr && mmap_data_ != MAP_FAILED) {
    munmap(mmap_data_, file_size_);
    mmap_data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

Result<void> CheckpointLoader::read_header() {
  if (file_size_ < sizeof(CheckpointHeader)) {
    return Err<void>(ErrorCode::FileReadError,
                    "Checkpoint file too small for header");
  }

  // Copy header from mmap
  std::memcpy(&header_, mmap_data_, sizeof(CheckpointHeader));

  // Validate header
  if (header_.dim <= 0 || header_.hidden_dim <= 0 || header_.n_layers <= 0 ||
      header_.n_heads <= 0 || header_.n_kv_heads <= 0 || header_.vocab_size <= 0 ||
      header_.seq_len <= 0) {
    return Err<void>(ErrorCode::FileReadError,
                    "Invalid checkpoint header: negative dimensions");
  }

  if (header_.n_heads % header_.n_kv_heads != 0) {
    return Err<void>(ErrorCode::FileReadError,
                    "Invalid checkpoint: n_heads must be divisible by n_kv_heads");
  }

  return Ok();
}

Result<void> CheckpointLoader::load_weights(LLaMAModel& model) const {
  // Create config from header
  TransformerConfig config;
  config.dim = header_.dim;
  config.hidden_dim = header_.hidden_dim;
  config.n_layers = header_.n_layers;
  config.n_heads = header_.n_heads;
  config.n_kv_heads = header_.n_kv_heads;
  config.vocab_size = header_.vocab_size;
  config.seq_len = header_.seq_len;
  config.head_size = header_.dim / header_.n_heads;  // Fixed: compute head_size
  config.norm_eps = 1e-5f;  // Standard epsilon for RMSNorm
  config.compute_derived();

  usize offset = 0;  // Current offset in float32 elements

  // ==== KuiperInfer weight order ====
  // 1. Embedding: [vocab_size × dim]
  // 2. Attention RMSNorm: [dim] × n_layers
  // 3. wq: [dim × dim] × n_layers
  // 4. wk: [kv_dim × dim] × n_layers
  // 5. wv: [kv_dim × dim] × n_layers
  // 6. wo: [dim × dim] × n_layers
  // 7. FFN RMSNorm: [dim] × n_layers
  // 8. w1: [hidden_dim × dim] × n_layers
  // 9. w2: [dim × hidden_dim] × n_layers
  // 10. w3: [hidden_dim × dim] × n_layers
  // 11. Final RMSNorm: [dim]
  // 12. Classifier: [vocab_size × dim] (if not shared)

  // Load weights in the correct KuiperInfer order:
  // 1. Embedding: [vocab_size × dim]
  // 2. Attention RMSNorm: [dim] × n_layers
  // 3. wq: [dim × dim] × n_layers
  // 4. wk: [kv_dim × dim] × n_layers
  // 5. wv: [kv_dim × dim] × n_layers
  // 6. wo: [dim × dim] × n_layers
  // 7. FFN RMSNorm: [dim] × n_layers
  // 8. w1: [hidden_dim × dim] × n_layers
  // 9. w2: [dim × hidden_dim] × n_layers
  // 10. w3: [hidden_dim × dim] × n_layers
  // 11. Final RMSNorm: [dim]
  // 12. Classifier: [vocab_size × dim] (if not shared)

  // 1. Load token embedding: [vocab_size × dim]
  {
    usize emb_size = static_cast<usize>(config.vocab_size) * static_cast<usize>(config.dim);
    auto emb_tensor = Tensor::create({config.vocab_size, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!emb_tensor) {
      return Err<void>(emb_tensor.error());
    }

    std::memcpy(emb_tensor.value().ptr<f32>(), weight_ptr(offset), emb_size * sizeof(f32));
    offset += emb_size;

    auto result = model.set_embedding(std::move(emb_tensor.value()));
    if (!result) return result;
  }

  // 2. Load attention RMSNorm weights: [dim] × n_layers
  std::vector<Tensor> attn_rmsnorm_weights;
  attn_rmsnorm_weights.reserve(config.n_layers);

  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto tensor = Tensor::create({config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), config.dim * sizeof(f32));
    offset += config.dim;
    attn_rmsnorm_weights.push_back(std::move(tensor.value()));
  }

  // 3-6. Load attention matmul weights: wq, wk, wv, wo
  std::vector<std::vector<Tensor>> attn_weights(4);  // wq, wk, wv, wo

  // wq: [dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    attn_weights[0].push_back(std::move(tensor.value()));
  }

  // wk: [kv_dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.kv_dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.kv_dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    attn_weights[1].push_back(std::move(tensor.value()));
  }

  // wv: [kv_dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.kv_dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.kv_dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    attn_weights[2].push_back(std::move(tensor.value()));
  }

  // wo: [dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    attn_weights[3].push_back(std::move(tensor.value()));
  }

  // 7. Load FFN RMSNorm weights: [dim] × n_layers
  std::vector<Tensor> ffn_rmsnorm_weights;
  ffn_rmsnorm_weights.reserve(config.n_layers);

  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto tensor = Tensor::create({config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), config.dim * sizeof(f32));
    offset += config.dim;
    ffn_rmsnorm_weights.push_back(std::move(tensor.value()));
  }

  // 8-10. Load FFN matmul weights: w1, w2, w3
  std::vector<std::vector<Tensor>> ffn_weights(3);  // w1, w2, w3

  // w1: [hidden_dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.hidden_dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.hidden_dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    ffn_weights[0].push_back(std::move(tensor.value()));
  }

  // w2: [dim × hidden_dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.dim) * static_cast<usize>(config.hidden_dim);
    auto tensor = Tensor::create({config.dim, config.hidden_dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    ffn_weights[1].push_back(std::move(tensor.value()));
  }

  // w3: [hidden_dim × dim] for each layer
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    usize size = static_cast<usize>(config.hidden_dim) * static_cast<usize>(config.dim);
    auto tensor = Tensor::create({config.hidden_dim, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), size * sizeof(f32));
    offset += size;
    ffn_weights[2].push_back(std::move(tensor.value()));
  }

  // 11. Load final RMSNorm: [dim]
  auto final_norm_tensor = Tensor::create({config.dim}, DataType::Float32, DeviceType::CPU);
  if (!final_norm_tensor) return Err<void>(final_norm_tensor.error());
  std::memcpy(final_norm_tensor.value().ptr<f32>(), weight_ptr(offset), config.dim * sizeof(f32));
  offset += config.dim;

  auto result = model.set_final_norm(std::move(final_norm_tensor.value()));
  if (!result) return result;

  // Now distribute weights to each transformer block
  for (i32 layer = 0; layer < config.n_layers; ++layer) {
    auto& block = model.get_block(layer);

    // Attention RMSNorm
    result = block.set_attn_norm(std::move(attn_rmsnorm_weights[layer]));
    if (!result) return result;

    // Attention weights
    result = block.set_wq(std::move(attn_weights[0][layer]));
    if (!result) return result;

    result = block.set_wk(std::move(attn_weights[1][layer]));
    if (!result) return result;

    result = block.set_wv(std::move(attn_weights[2][layer]));
    if (!result) return result;

    result = block.set_wo(std::move(attn_weights[3][layer]));
    if (!result) return result;

    // FFN RMSNorm
    result = block.set_ffn_norm(std::move(ffn_rmsnorm_weights[layer]));
    if (!result) return result;

    // FFN weights
    result = block.set_w1(std::move(ffn_weights[0][layer]));
    if (!result) return result;

    result = block.set_w2(std::move(ffn_weights[1][layer]));
    if (!result) return result;

    result = block.set_w3(std::move(ffn_weights[2][layer]));
    if (!result) return result;
  }

  // NOTE: freq_cis is NOT in the checkpoint file!
  // KuiperInfer computes RoPE cache at runtime, not from the file.
  // So we don't skip any bytes here.

  // Classifier: check if separate weights exist
  usize bytes_remaining = file_size_ - sizeof(CheckpointHeader) - (offset * sizeof(f32));
  usize classifier_size = static_cast<usize>(config.vocab_size) * static_cast<usize>(config.dim) * sizeof(f32);

  if (bytes_remaining >= classifier_size) {
    // Separate classifier weights
    auto tensor = Tensor::create({config.vocab_size, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(offset), classifier_size);
    offset += (classifier_size / sizeof(f32));
    auto result = model.set_classifier(std::move(tensor.value()));
    if (!result) return result;
  } else {
    // Weight sharing: use embedding weights for classifier
    auto tensor = Tensor::create({config.vocab_size, config.dim}, DataType::Float32, DeviceType::CPU);
    if (!tensor) return Err<void>(tensor.error());
    // Copy from embedding weights (at offset 0)
    std::memcpy(tensor.value().ptr<f32>(), weight_ptr(0),
                static_cast<usize>(config.vocab_size) * static_cast<usize>(config.dim) * sizeof(f32));
    auto result = model.set_classifier(std::move(tensor.value()));
    if (!result) return result;
  }

  return Ok();
}

}  // namespace photon::model
