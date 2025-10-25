/**
 * @file llama_model.cpp
 * @brief LLaMA model implementation
 * @version 0.1.0
 */

#include "photon/model/llama_model.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace photon::model {

LLaMAModel::LLaMAModel(const TransformerConfig& config)
    : config_(config),
      embedding_(config.vocab_size, config.dim),
      final_norm_(config.dim, config.norm_eps),
      classifier_(config.dim, config.vocab_size) {  // Fixed: input=dim, output=vocab_size

  // Set device for all operators
  embedding_.set_device(config.device);
  final_norm_.set_device(config.device);
  classifier_.set_device(config.device);

  // Create transformer blocks
  blocks_.reserve(config.n_layers);
  for (i32 i = 0; i < config.n_layers; ++i) {
    blocks_.push_back(std::make_unique<TransformerBlock>(i, config));
  }
}

Result<void> LLaMAModel::init() {
  if (initialized_) {
    return Ok();
  }

  // NOTE: Parameterized operators (embedding, final_norm, classifier) should have
  // their weights set before calling this. They are initialized via set_weight() calls.

  // Initialize all transformer blocks (this allocates their intermediate buffers)
  for (auto& block : blocks_) {
    auto block_init = block->init();
    if (!block_init) return block_init;
  }

  // Allocate KV cache for each layer
  auto device = config_.device;
  auto dtype = DataType::Float32;

  key_cache_.clear();
  value_cache_.clear();
  key_cache_.reserve(config_.n_layers);
  value_cache_.reserve(config_.n_layers);

  for (i32 i = 0; i < config_.n_layers; ++i) {
    auto key_result = Tensor::create({config_.seq_len, config_.kv_dim}, dtype, device);
    if (!key_result) {
      return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate key cache");
    }
    key_cache_.push_back(std::move(key_result.value()));

    auto value_result = Tensor::create({config_.seq_len, config_.kv_dim}, dtype, device);
    if (!value_result) {
      return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate value cache");
    }
    value_cache_.push_back(std::move(value_result.value()));
  }

  // Allocate working buffers
  auto x_result = Tensor::create({config_.dim}, dtype, device);
  if (!x_result) return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate x buffer");
  x_ = std::move(x_result.value());

  auto emb_result = Tensor::create({1, config_.dim}, dtype, device);  // Fixed: 2D output
  if (!emb_result) return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate emb buffer");
  emb_out_ = std::move(emb_result.value());

  auto norm_result = Tensor::create({config_.dim}, dtype, device);
  if (!norm_result) return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate norm buffer");
  norm_out_ = std::move(norm_result.value());

  auto logits_result = Tensor::create({config_.vocab_size}, dtype, device);
  if (!logits_result) return Err<void>(ErrorCode::OutOfMemory, "Failed to allocate logits buffer");
  logits_buf_ = std::move(logits_result.value());

  initialized_ = true;
  return Ok();
}

Result<void> LLaMAModel::set_embedding(Tensor weight) {
  // Convert to model's device if needed
  if (weight.device() != config_.device) {
    auto converted = weight.to(config_.device);
    if (!converted) return Err<void>(converted.error());
    weight = std::move(converted.value());
  }

  auto result = embedding_.set_weight(std::move(weight));
  if (!result) return result;
  return embedding_.init();
}

Result<void> LLaMAModel::set_final_norm(Tensor weight) {
  // Convert to model's device if needed
  if (weight.device() != config_.device) {
    auto converted = weight.to(config_.device);
    if (!converted) return Err<void>(converted.error());
    weight = std::move(converted.value());
  }

  auto result = final_norm_.set_weight(std::move(weight));
  if (!result) return result;
  return final_norm_.init();
}

Result<void> LLaMAModel::set_classifier(Tensor weight) {
  // Convert to model's device if needed
  if (weight.device() != config_.device) {
    auto converted = weight.to(config_.device);
    if (!converted) return Err<void>(converted.error());
    weight = std::move(converted.value());
  }

  auto result = classifier_.set_weight(std::move(weight));
  if (!result) return result;
  return classifier_.init();
}

TransformerBlock& LLaMAModel::get_block(i32 layer_idx) {
  return *blocks_[layer_idx];
}

Result<void> LLaMAModel::forward(i32 token, i32 pos, Tensor& logits) {
  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator, "LLaMAModel not initialized");
  }

  if (pos < 0 || pos >= config_.seq_len) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Position out of bounds: " + std::to_string(pos));
  }

  // 1. Embedding lookup
  // Create a temporary tensor with the single token (on correct device)
  auto token_tensor = Tensor::create({1}, DataType::Int32, config_.device);
  if (!token_tensor) {
    return Err<void>(token_tensor.error());
  }

  // For CUDA, we need to copy token from CPU to GPU
  if (config_.device == DeviceType::CUDA) {
    i32 token_cpu = token;
    cudaError_t err = cudaMemcpy(token_tensor.value().data(), &token_cpu,
                                 sizeof(i32), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      std::string("Failed to copy token to GPU: ") +
                      cudaGetErrorString(err));
    }
  } else {
    i32* token_ptr = token_tensor.value().ptr<i32>();
    token_ptr[0] = token;
  }

  auto emb_result = embedding_.forward(token_tensor.value(), emb_out_);
  if (!emb_result) return emb_result;

  // Copy embedding to x_ (emb_out_ is [1 × dim], x_ is [dim])
  if (config_.device == DeviceType::CUDA) {
    // Use cudaMemcpy for GPU tensors
    cudaError_t err = cudaMemcpy(x_.data(), emb_out_.data(),
                                 config_.dim * sizeof(f32),
                                 cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      std::string("Failed to copy embedding to x: ") +
                      cudaGetErrorString(err));
    }
  } else {
    // CPU: direct pointer access
    f32* emb_ptr = emb_out_.ptr<f32>();
    f32* x_ptr = x_.ptr<f32>();
    for (i32 i = 0; i < config_.dim; ++i) {
      x_ptr[i] = emb_ptr[i];
    }
  }

  // 2. Forward through all transformer blocks
  for (i32 layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    auto block_result = blocks_[layer_idx]->forward(
        x_, pos, key_cache_[layer_idx], value_cache_[layer_idx]);
    if (!block_result) return block_result;
  }

  // 3. Final RMSNorm
  auto norm_result = final_norm_.forward(x_, norm_out_);
  if (!norm_result) return norm_result;

  // 4. Classifier projection
  // If logits is on a different device, use internal buffer and copy
  if (logits.device() == config_.device) {
    // Same device, direct output
    auto cls_result = classifier_.forward(norm_out_, logits);
    if (!cls_result) return cls_result;
  } else {
    // Different device, use internal buffer then copy
    auto cls_result = classifier_.forward(norm_out_, logits_buf_);
    if (!cls_result) return cls_result;

    // Copy from CUDA to CPU (or vice versa)
    auto converted = logits_buf_.to(logits.device());
    if (!converted) return Err<void>(converted.error());

    // Copy data to output tensor
    std::memcpy(logits.data(), converted.value().data(), logits.byte_size());
  }

  return Ok();
}

Result<i32> LLaMAModel::generate_next(const std::vector<i32>& tokens) {
  if (tokens.empty()) {
    return Err<i32>(ErrorCode::InvalidArgument, "Empty token sequence");
  }

  // Process all tokens to fill KV cache
  for (usize i = 0; i < tokens.size(); ++i) {
    auto fwd_result = forward(tokens[i], static_cast<i32>(i), logits_buf_);
    if (!fwd_result) {
      return Err<i32>(fwd_result.error());
    }
  }

  // Sample next token using argmax
  return argmax_sample(logits_buf_);
}

void LLaMAModel::reset_cache() {
  // Zero out all KV caches
  for (auto& key_cache : key_cache_) {
    f32* ptr = key_cache.ptr<f32>();
    std::fill(ptr, ptr + key_cache.size(), 0.0f);
  }

  for (auto& value_cache : value_cache_) {
    f32* ptr = value_cache.ptr<f32>();
    std::fill(ptr, ptr + value_cache.size(), 0.0f);
  }
}

Result<i32> argmax_sample(const Tensor& logits) {
  if (logits.empty()) {
    return Err<i32>(ErrorCode::InvalidArgument, "Empty logits tensor");
  }

  if (logits.dtype() != DataType::Float32) {
    return Err<i32>(ErrorCode::InvalidDtype, "Logits must be Float32");
  }

  const f32* logits_ptr = logits.ptr<f32>();
  i32 vocab_size = static_cast<i32>(logits.size());

  // Find index of maximum logit
  i32 max_idx = 0;
  f32 max_val = logits_ptr[0];

  for (i32 i = 1; i < vocab_size; ++i) {
    if (logits_ptr[i] > max_val) {
      max_val = logits_ptr[i];
      max_idx = i;
    }
  }

  return Ok(max_idx);
}

Result<void> LLaMAModel::quantize_weights(i32 group_size) {
  std::cout << "Quantizing model weights to INT8 (group_size=" << group_size << ")..." << std::endl;

  // Quantize all transformer blocks
  for (usize i = 0; i < blocks_.size(); ++i) {
    std::cout << "  Layer " << i << "/" << blocks_.size() << "..." << std::flush;
    auto result = blocks_[i]->quantize_weights(group_size);
    if (!result) {
      return Err<void>(result.error().code(),
                      "Failed to quantize layer " + std::to_string(i) + ": " +
                      result.error().message());
    }
    std::cout << " done" << std::endl;
  }

  // Quantize classifier
  std::cout << "  Classifier..." << std::flush;
  auto classifier_result = classifier_.quantize_weight(group_size);
  if (!classifier_result) {
    return Err<void>(classifier_result.error().code(),
                    "Failed to quantize classifier: " +
                    classifier_result.error().message());
  }
  std::cout << " done" << std::endl;

  std::cout << "✓ Quantization complete! Model memory reduced by ~3.8x" << std::endl;

  return Ok();
}

}  // namespace photon::model
