/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file llama_model.cpp
 * @brief LLaMA model implementation
 * @version 0.1.0
 */

#include "photon/arch/llama_model.hpp"
#include "photon/runtime/kv_cache_manager.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <glog/logging.h>

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

// Destructor must be defined in .cpp where KVCacheManager is complete
LLaMAModel::~LLaMAModel() {
  // Free batched GPU buffers
#ifdef PHOTON_USE_CUDA
  if (positions_gpu_) cudaFree(positions_gpu_);
  if (seq_ids_gpu_) cudaFree(seq_ids_gpu_);
  if (cache_offsets_gpu_) cudaFree(cache_offsets_gpu_);

  // Destroy cuBLAS handle
  if (cublas_handle_ != nullptr) {
    cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
    cublas_handle_ = nullptr;
  }
#endif
}

Result<void> LLaMAModel::init() {
  if (initialized_) {
    return Ok();
  }

  // NOTE: Parameterized operators (embedding, final_norm, classifier) should have
  // their weights set before calling this. They are initialized via set_weight() calls.

#ifdef PHOTON_USE_CUDA
  // Create cuBLAS handle for FP16 Tensor Core optimization
  if (config_.device == DeviceType::CUDA && cublas_handle_ == nullptr) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return Err<void>(ErrorCode::CudaError,
                      "Failed to create cuBLAS handle: " + std::to_string(status));
    }
    cublas_handle_ = static_cast<void*>(handle);

    // Set cuBLAS handle for classifier
    classifier_.set_cublas_handle(cublas_handle_);

    // Set cuBLAS handle for all transformer blocks
    for (auto& block : blocks_) {
      block->set_cublas_handle(cublas_handle_);
    }
  }
#endif

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

// ============================================================================
// Batched Inference Support
// ============================================================================

Result<void> LLaMAModel::ensure_batched_buffers(i32 batch_size) {
  if (batch_size <= batched_buffers_capacity_) {
    return Ok();  // Already allocated
  }

#ifdef PHOTON_USE_CUDA
  // Free old buffers if they exist
  if (positions_gpu_) cudaFree(positions_gpu_);
  if (seq_ids_gpu_) cudaFree(seq_ids_gpu_);
  if (cache_offsets_gpu_) cudaFree(cache_offsets_gpu_);

  // Allocate new GPU buffers
  cudaError_t err;
  err = cudaMalloc(&positions_gpu_, batch_size * sizeof(i32));
  if (err != cudaSuccess) {
    return Err<void>(ErrorCode::CudaError,
                    std::string("Failed to allocate positions_gpu: ") +
                    cudaGetErrorString(err));
  }

  err = cudaMalloc(&seq_ids_gpu_, batch_size * sizeof(i32));
  if (err != cudaSuccess) {
    cudaFree(positions_gpu_);
    positions_gpu_ = nullptr;
    return Err<void>(ErrorCode::CudaError,
                    std::string("Failed to allocate seq_ids_gpu: ") +
                    cudaGetErrorString(err));
  }

  err = cudaMalloc(&cache_offsets_gpu_, batch_size * sizeof(i32));
  if (err != cudaSuccess) {
    cudaFree(positions_gpu_);
    cudaFree(seq_ids_gpu_);
    positions_gpu_ = nullptr;
    seq_ids_gpu_ = nullptr;
    return Err<void>(ErrorCode::CudaError,
                    std::string("Failed to allocate cache_offsets_gpu: ") +
                    cudaGetErrorString(err));
  }
#endif

  // Allocate Tensor buffers
  auto tokens_result = Tensor::create({batch_size}, DataType::Int32, DeviceType::CUDA);
  if (!tokens_result) {
#ifdef PHOTON_USE_CUDA
    cudaFree(positions_gpu_);
    cudaFree(seq_ids_gpu_);
    cudaFree(cache_offsets_gpu_);
    positions_gpu_ = nullptr;
    seq_ids_gpu_ = nullptr;
    cache_offsets_gpu_ = nullptr;
#endif
    return Err<void>(tokens_result.error());
  }
  tokens_batch_ = std::move(tokens_result.value());

  auto x_result = Tensor::create({batch_size, config_.dim}, DataType::Float32, DeviceType::CUDA);
  if (!x_result) {
#ifdef PHOTON_USE_CUDA
    cudaFree(positions_gpu_);
    cudaFree(seq_ids_gpu_);
    cudaFree(cache_offsets_gpu_);
    positions_gpu_ = nullptr;
    seq_ids_gpu_ = nullptr;
    cache_offsets_gpu_ = nullptr;
#endif
    return Err<void>(x_result.error());
  }
  x_batch_ = std::move(x_result.value());

  auto norm_result = Tensor::create({batch_size, config_.dim}, DataType::Float32, DeviceType::CUDA);
  if (!norm_result) {
#ifdef PHOTON_USE_CUDA
    cudaFree(positions_gpu_);
    cudaFree(seq_ids_gpu_);
    cudaFree(cache_offsets_gpu_);
    positions_gpu_ = nullptr;
    seq_ids_gpu_ = nullptr;
    cache_offsets_gpu_ = nullptr;
#endif
    return Err<void>(norm_result.error());
  }
  norm_batch_ = std::move(norm_result.value());

  batched_buffers_capacity_ = batch_size;
  LOG(INFO) << "Allocated batched buffers for batch_size=" << batch_size;
  return Ok();
}

Result<void> LLaMAModel::init_paged_cache(i32 num_blocks, i32 block_size) {
  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "Model must be initialized before creating paged cache");
  }

  if (config_.device != DeviceType::CUDA) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Paged cache currently only supported on CUDA");
  }

  cache_manager_ = std::make_unique<KVCacheManager>(
      num_blocks, block_size, config_.n_layers, config_.kv_dim, config_.device);

  auto init_result = cache_manager_->init();
  if (!init_result) {
    return Err<void>(init_result.error());
  }

  use_paged_cache_ = true;
  LOG(INFO) << "Paged KV cache initialized successfully";
  return Ok();
}

Result<void> LLaMAModel::forward_batched(
    const std::vector<i32>& tokens,
    const std::vector<i32>& positions,
    const std::vector<i32>& seq_ids,
    Tensor& logits) {

  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator, "Model not initialized");
  }

  if (!use_paged_cache_ || !cache_manager_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "Paged cache not initialized. Call init_paged_cache() first");
  }

  if (config_.device != DeviceType::CUDA) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "Batched forward currently only supported on CUDA");
  }

  const i32 batch_size = static_cast<i32>(tokens.size());
  if (batch_size == 0) {
    return Err<void>(ErrorCode::InvalidArgument, "Empty batch");
  }

  if (positions.size() != static_cast<usize>(batch_size) ||
      seq_ids.size() != static_cast<usize>(batch_size)) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "tokens, positions, and seq_ids must have same size");
  }

  if (logits.ndim() != 2 ||
      logits.dims()[0] != batch_size ||
      logits.dims()[1] != config_.vocab_size) {
    return Err<void>(ErrorCode::InvalidShape,
                    "logits must have shape [batch_size, vocab_size]");
  }

  if (logits.device() != DeviceType::CUDA) {
    return Err<void>(ErrorCode::InvalidArgument,
                    "logits tensor must be on CUDA for batched forward");
  }

  // ========================================================================
  // TRUE BATCHED PROCESSING - Key optimization for 250+ tokens/s!
  // ========================================================================

  // Ensure pre-allocated buffers are ready
  auto buffer_result = ensure_batched_buffers(batch_size);
  if (!buffer_result) return buffer_result;

  // Copy tokens to GPU (reusing pre-allocated buffer)
  cudaMemcpy(tokens_batch_.data(), tokens.data(),
             batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  // Copy positions to GPU (reusing pre-allocated buffer)
  cudaMemcpy(positions_gpu_, positions.data(),
             batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  // Copy seq_ids to GPU (reusing pre-allocated buffer)
  cudaMemcpy(seq_ids_gpu_, seq_ids.data(),
             batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  // Prepare cache_offsets once for all layers (optimization!)
  std::vector<i32> cache_offsets_cpu(batch_size);
  for (i32 i = 0; i32 seq_id : seq_ids) {
    auto offset_result = cache_manager_->get_sequence_offset(seq_id);
    if (!offset_result) {
      return Err<void>(offset_result.error());
    }
    cache_offsets_cpu[i++] = offset_result.value();
  }
  cudaMemcpy(cache_offsets_gpu_, cache_offsets_cpu.data(),
             batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  // 1. Batched Embedding Lookup (writes directly to x_batch_)
  auto emb_result = embedding_.forward(tokens_batch_, x_batch_);
  if (!emb_result) return emb_result;

  // 2. Forward through all transformer blocks (BATCHED!)
  for (i32 layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    Tensor& key_cache = cache_manager_->get_key_cache(layer_idx);
    Tensor& value_cache = cache_manager_->get_value_cache(layer_idx);

    // Call BATCHED forward with proper cache management
    // Pass both GPU pointers (for kernel use) and CPU vectors (for cache management)
    auto block_result = blocks_[layer_idx]->forward_batched(
        x_batch_, positions_gpu_, positions, seq_ids, cache_offsets_gpu_, batch_size,
        key_cache, value_cache, cache_manager_.get());

    if (!block_result) return block_result;
  }

  // 3. Batched Final RMSNorm (using pre-allocated norm_batch_)
  auto norm_result = final_norm_.forward(x_batch_, norm_batch_);
  if (!norm_result) return norm_result;

  // 4. Batched Classifier projection
  auto cls_result = classifier_.forward(norm_batch_, logits);
  if (!cls_result) return cls_result;

  return Ok();
}

}  // namespace photon::model
