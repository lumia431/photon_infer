/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file transformer_block.cpp
 * @brief Transformer block implementation
 * @version 0.1.0
 */

#include "photon/arch/transformer_block.hpp"
#include "photon/runtime/kv_cache_manager.hpp"

#include <glog/logging.h>

#ifdef PHOTON_USE_CUDA
#include <cuda_runtime.h>
#include "photon/ops/kernels/cuda/batched_mha_kernel.cuh"
#include "photon/ops/kernels/cuda/kv_cache_kernel.cuh"
#endif

namespace photon::model {

TransformerBlock::TransformerBlock(i32 layer_idx, const TransformerConfig& config)
    : layer_idx_(layer_idx),
      config_(config),
      attn_norm_(config.dim, config.norm_eps),
      ffn_norm_(config.dim, config.norm_eps),
      wq_(config.dim, config.dim),
      wk_(config.dim, config.kv_dim),   // Fixed: input=dim, output=kv_dim
      wv_(config.dim, config.kv_dim),   // Fixed: input=dim, output=kv_dim
      wo_(config.dim, config.dim),
      w1_(config.dim, config.hidden_dim),  // Fixed: input=dim, output=hidden_dim
      w2_(config.hidden_dim, config.dim),
      w3_(config.dim, config.hidden_dim),  // Fixed: input=dim, output=hidden_dim
      rope_(config.dim, config.kv_dim, config.head_size, config.seq_len),
      mha_(config.dim, config.kv_dim, config.n_heads, config.head_size, config.seq_len),
      add_(),
      swiglu_(config.hidden_dim) {
  // Set device for all operators
  attn_norm_.set_device(config.device);
  ffn_norm_.set_device(config.device);
  wq_.set_device(config.device);
  wk_.set_device(config.device);
  wv_.set_device(config.device);
  wo_.set_device(config.device);
  w1_.set_device(config.device);
  w2_.set_device(config.device);
  w3_.set_device(config.device);
  rope_.set_device(config.device);
  mha_.set_device(config.device);
  add_.set_device(config.device);
  swiglu_.set_device(config.device);
}

Result<void> TransformerBlock::init() {
  if (initialized_) {
    return Ok();
  }

  // NOTE: We do NOT call init() on parameterized operators (attn_norm, ffn_norm,
  // wq, wk, wv, wo, w1, w2, w3) here because they require weights to be set first.
  // Their init() will be called when set_weight() is called on them.

  // Initialize non-parameterized operators
  auto rope_init = rope_.init();
  if (!rope_init) return rope_init;

  auto mha_init = mha_.init();
  if (!mha_init) return mha_init;

  auto add_init = add_.init();
  if (!add_init) return add_init;

  auto swiglu_init = swiglu_.init();
  if (!swiglu_init) return swiglu_init;

  // Allocate intermediate buffers
  auto device = config_.device;
  auto dtype = DataType::Float32;

  auto create_buf = [&](i32 size) -> Result<Tensor> {
    return Tensor::create({size}, dtype, device);
  };

  auto r1 = create_buf(config_.dim);
  if (!r1) return Err<void>(r1.error());
  attn_out_ = std::move(r1.value());

  auto r2 = create_buf(config_.dim);
  if (!r2) return Err<void>(r2.error());
  q_ = std::move(r2.value());

  auto r3 = create_buf(config_.kv_dim);
  if (!r3) return Err<void>(r3.error());
  k_ = std::move(r3.value());

  auto r4 = create_buf(config_.kv_dim);
  if (!r4) return Err<void>(r4.error());
  v_ = std::move(r4.value());

  auto r5 = create_buf(config_.dim);
  if (!r5) return Err<void>(r5.error());
  attn_result_ = std::move(r5.value());

  auto r6 = create_buf(config_.dim);
  if (!r6) return Err<void>(r6.error());
  wo_out_ = std::move(r6.value());

  auto r7 = create_buf(config_.dim);
  if (!r7) return Err<void>(r7.error());
  ffn_out_ = std::move(r7.value());

  auto r8 = create_buf(config_.hidden_dim);
  if (!r8) return Err<void>(r8.error());
  w1_out_ = std::move(r8.value());

  auto r9 = create_buf(config_.hidden_dim);
  if (!r9) return Err<void>(r9.error());
  w3_out_ = std::move(r9.value());

  auto r10 = create_buf(config_.hidden_dim);
  if (!r10) return Err<void>(r10.error());
  swiglu_out_ = std::move(r10.value());

  auto r11 = create_buf(config_.dim);
  if (!r11) return Err<void>(r11.error());
  w2_out_ = std::move(r11.value());

  initialized_ = true;
  return Ok();
}

// Helper function to convert weight to correct device
template<typename OpType>
Result<void> set_weight_with_device_conversion(
    OpType& op, Tensor weight, DeviceType target_device) {
  if (weight.device() != target_device) {
    auto converted = weight.to(target_device);
    if (!converted) return Err<void>(converted.error());
    weight = std::move(converted.value());
  }

  auto result = op.set_weight(std::move(weight));
  if (!result) return result;
  return op.init();
}

Result<void> TransformerBlock::set_wq(Tensor weight) {
  return set_weight_with_device_conversion(wq_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_wk(Tensor weight) {
  return set_weight_with_device_conversion(wk_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_wv(Tensor weight) {
  return set_weight_with_device_conversion(wv_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_wo(Tensor weight) {
  return set_weight_with_device_conversion(wo_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_w1(Tensor weight) {
  return set_weight_with_device_conversion(w1_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_w2(Tensor weight) {
  return set_weight_with_device_conversion(w2_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_w3(Tensor weight) {
  return set_weight_with_device_conversion(w3_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_attn_norm(Tensor weight) {
  return set_weight_with_device_conversion(attn_norm_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::set_ffn_norm(Tensor weight) {
  return set_weight_with_device_conversion(ffn_norm_, std::move(weight), config_.device);
}

Result<void> TransformerBlock::forward(Tensor& x, i32 pos,
                                      Tensor& key_cache, Tensor& value_cache) {
  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator, "TransformerBlock not initialized");
  }

  // ========================================================================
  // Self-Attention Block
  // ========================================================================

  // 1. Pre-attention RMSNorm
  auto norm_result = attn_norm_.forward(x, attn_out_);
  if (!norm_result) return norm_result;

  // 2. Project to Q, K, V
  auto wq_result = wq_.forward(attn_out_, q_);
  if (!wq_result) return wq_result;

  auto wk_result = wk_.forward(attn_out_, k_);
  if (!wk_result) return wk_result;

  auto wv_result = wv_.forward(attn_out_, v_);
  if (!wv_result) return wv_result;

  // 3. Apply RoPE to Q and K
  auto rope_result = rope_.forward(q_, k_, pos);
  if (!rope_result) return rope_result;

  // 4. Store K, V into cache at position pos
  // key_cache shape: [seq_len × kv_dim]
  // Copy k_ into key_cache[pos, :]
  if (config_.device == DeviceType::CUDA) {
#ifdef PHOTON_USE_CUDA
    // Use cudaMemcpy for GPU tensors
    f32* k_src = k_.ptr<f32>();
    f32* key_cache_dst = key_cache.ptr<f32>() + pos * config_.kv_dim;
    cudaError_t err = cudaMemcpy(key_cache_dst, k_src,
                                 config_.kv_dim * sizeof(f32),
                                 cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      std::string("Failed to copy K to cache: ") +
                      cudaGetErrorString(err));
    }

    // Copy V to cache
    f32* v_src = v_.ptr<f32>();
    f32* value_cache_dst = value_cache.ptr<f32>() + pos * config_.kv_dim;
    err = cudaMemcpy(value_cache_dst, v_src,
                    config_.kv_dim * sizeof(f32),
                    cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      return Err<void>(ErrorCode::CudaError,
                      std::string("Failed to copy V to cache: ") +
                      cudaGetErrorString(err));
    }
#endif
  } else {
    // CPU: direct pointer access
    f32* k_ptr = k_.ptr<f32>();
    f32* key_cache_ptr = key_cache.ptr<f32>();
    for (i32 i = 0; i < config_.kv_dim; ++i) {
      key_cache_ptr[pos * config_.kv_dim + i] = k_ptr[i];
    }

    f32* v_ptr = v_.ptr<f32>();
    f32* value_cache_ptr = value_cache.ptr<f32>();
    for (i32 i = 0; i < config_.kv_dim; ++i) {
      value_cache_ptr[pos * config_.kv_dim + i] = v_ptr[i];
    }
  }

  // 5. Multi-head attention
  auto mha_result = mha_.forward(q_, key_cache, value_cache, attn_result_, pos);
  if (!mha_result) return mha_result;

  // 6. Output projection
  auto wo_result = wo_.forward(attn_result_, wo_out_);
  if (!wo_result) return wo_result;

  // 7. Residual connection: x = x + wo_out
  auto add1_result = add_.forward(x, wo_out_, x);  // In-place
  if (!add1_result) return add1_result;

  // ========================================================================
  // Feed-Forward Block
  // ========================================================================

  // 8. Pre-FFN RMSNorm
  auto ffn_norm_result = ffn_norm_.forward(x, ffn_out_);
  if (!ffn_norm_result) return ffn_norm_result;

  // 9. FFN: w2(swiglu(w1(h), w3(h)))
  auto w1_result = w1_.forward(ffn_out_, w1_out_);
  if (!w1_result) return w1_result;

  auto w3_result = w3_.forward(ffn_out_, w3_out_);
  if (!w3_result) return w3_result;

  auto swiglu_result = swiglu_.forward(w1_out_, w3_out_, swiglu_out_);
  if (!swiglu_result) return swiglu_result;

  auto w2_result = w2_.forward(swiglu_out_, w2_out_);
  if (!w2_result) return w2_result;

  // 10. Residual connection: x = x + w2_out
  auto add2_result = add_.forward(x, w2_out_, x);  // In-place
  if (!add2_result) return add2_result;

  return Ok();
}

// ============================================================================
// Batched Forward Pass
// ============================================================================

Result<void> TransformerBlock::ensure_batch_buffers(i32 batch_size) {
  if (batch_size <= current_batch_capacity_) {
    return Ok();  // Already allocated
  }

  auto device = config_.device;
  auto dtype = DataType::Float32;

  // Helper to create [batch, size] tensor
  auto create_batch_buf = [&](i32 size) -> Result<Tensor> {
    return Tensor::create({batch_size, size}, dtype, device);
  };

  // Allocate all batched buffers
  auto r1 = create_batch_buf(config_.dim);
  if (!r1) return Err<void>(r1.error());
  attn_out_batch_ = std::move(r1.value());

  auto r2 = create_batch_buf(config_.dim);
  if (!r2) return Err<void>(r2.error());
  q_batch_ = std::move(r2.value());

  auto r3 = create_batch_buf(config_.kv_dim);
  if (!r3) return Err<void>(r3.error());
  k_batch_ = std::move(r3.value());

  auto r4 = create_batch_buf(config_.kv_dim);
  if (!r4) return Err<void>(r4.error());
  v_batch_ = std::move(r4.value());

  auto r5 = create_batch_buf(config_.dim);
  if (!r5) return Err<void>(r5.error());
  attn_result_batch_ = std::move(r5.value());

  auto r6 = create_batch_buf(config_.dim);
  if (!r6) return Err<void>(r6.error());
  wo_out_batch_ = std::move(r6.value());

  auto r7 = create_batch_buf(config_.dim);
  if (!r7) return Err<void>(r7.error());
  ffn_out_batch_ = std::move(r7.value());

  auto r8 = create_batch_buf(config_.hidden_dim);
  if (!r8) return Err<void>(r8.error());
  w1_out_batch_ = std::move(r8.value());

  auto r9 = create_batch_buf(config_.hidden_dim);
  if (!r9) return Err<void>(r9.error());
  w3_out_batch_ = std::move(r9.value());

  auto r10 = create_batch_buf(config_.hidden_dim);
  if (!r10) return Err<void>(r10.error());
  swiglu_out_batch_ = std::move(r10.value());

  auto r11 = create_batch_buf(config_.dim);
  if (!r11) return Err<void>(r11.error());
  w2_out_batch_ = std::move(r11.value());

  // Allocate temporary cache buffers (one-time allocation, reused across sequences)
  if (temp_key_cache_.empty()) {
    auto key_result = Tensor::create({config_.seq_len, config_.kv_dim}, dtype, device);
    if (!key_result) return Err<void>(key_result.error());
    temp_key_cache_ = std::move(key_result.value());
  }

  if (temp_value_cache_.empty()) {
    auto value_result = Tensor::create({config_.seq_len, config_.kv_dim}, dtype, device);
    if (!value_result) return Err<void>(value_result.error());
    temp_value_cache_ = std::move(value_result.value());
  }

  // Allocate cached GPU buffers for batched MHA (device-only, reused)
  if (config_.device == DeviceType::CUDA) {
    // Allocate cache_offsets buffer on GPU
    auto offsets_result = Tensor::create({batch_size}, DataType::Int32, DeviceType::CUDA);
    if (!offsets_result) return Err<void>(offsets_result.error());
    cached_offsets_gpu_ = std::move(offsets_result.value());

    // Allocate score buffer on GPU
    auto score_result = Tensor::create({batch_size, config_.n_heads, config_.seq_len},
                                       DataType::Float32, DeviceType::CUDA);
    if (!score_result) return Err<void>(score_result.error());
    cached_score_buf_ = std::move(score_result.value());
  }

  current_batch_capacity_ = batch_size;
  return Ok();
}

Result<void> TransformerBlock::forward_batched(
    Tensor& x_batch,
    const i32* positions_gpu,
    const std::vector<i32>& positions_cpu,
    const std::vector<i32>& seq_ids_cpu,
    const i32* cache_offsets_gpu,
    i32 batch_size,
    Tensor& key_cache,
    Tensor& value_cache,
    KVCacheManager* cache_manager) {

  if (!initialized_) {
    return Err<void>(ErrorCode::InvalidOperator,
                    "TransformerBlock not initialized");
  }

  // Ensure batch buffers are allocated
  auto alloc_result = ensure_batch_buffers(batch_size);
  if (!alloc_result) return alloc_result;

  // Verify input shape
  if (x_batch.ndim() != 2 || x_batch.dims()[0] != batch_size ||
      x_batch.dims()[1] != config_.dim) {
    return Err<void>(ErrorCode::InvalidShape,
                    "x_batch must have shape [batch_size, dim]");
  }

  // ========================================================================
  // Attention Block
  // ========================================================================

  // 1. Pre-attention RMSNorm (batched)
  auto attn_norm_result = attn_norm_.forward(x_batch, attn_out_batch_);
  if (!attn_norm_result) return attn_norm_result;

  // 2. Q, K, V projections (batched MatMul)
  auto wq_result = wq_.forward(attn_out_batch_, q_batch_);
  if (!wq_result) return wq_result;

  auto wk_result = wk_.forward(attn_out_batch_, k_batch_);
  if (!wk_result) return wk_result;

  auto wv_result = wv_.forward(attn_out_batch_, v_batch_);
  if (!wv_result) return wv_result;

  // 3. Apply RoPE - use first position from CPU (no GPU→CPU copy!)
  i32 pos = positions_cpu[0];

  // Apply RoPE to entire batch at once
  auto rope_result = rope_.forward(q_batch_, k_batch_, pos);
  if (!rope_result) return rope_result;

  // Store K, V to cache (batched GPU kernel for efficiency)
  if (config_.device == DeviceType::CUDA) {
#ifdef PHOTON_USE_CUDA
    // Use optimized batched kernel to write all K/V pairs in one kernel call
    // This replaces batch_size * 2 cudaMemcpy calls with a single kernel launch
    auto kv_write_result = photon::kernels::cuda::batched_kv_write_launch(
        k_batch_.ptr<f32>(),      // Source K [batch_size, kv_dim]
        v_batch_.ptr<f32>(),      // Source V [batch_size, kv_dim]
        key_cache.ptr<f32>(),     // Destination key cache
        value_cache.ptr<f32>(),   // Destination value cache
        cache_offsets_gpu,        // Cache offsets [batch_size] (GPU)
        positions_gpu,            // Positions [batch_size] (GPU)
        batch_size,
        config_.kv_dim,
        nullptr);                 // Use default stream

    if (!kv_write_result) {
      return Err<void>(kv_write_result.error());
    }
#endif
  } else {
    // CPU fallback: loop through batch
    for (i32 i = 0; i < batch_size; ++i) {
      i32 seq_id = seq_ids_cpu[i];
      i32 seq_pos = positions_cpu[i];

      auto offset_result = cache_manager->get_sequence_offset(seq_id);
      if (!offset_result) {
        return Err<void>(offset_result.error());
      }
      i32 seq_offset = offset_result.value();
      i32 cache_idx = seq_offset + seq_pos;

      f32* k_batch_ptr = k_batch_.ptr<f32>() + i * config_.kv_dim;
      f32* v_batch_ptr = v_batch_.ptr<f32>() + i * config_.kv_dim;
      f32* key_cache_ptr = key_cache.ptr<f32>();
      f32* value_cache_ptr = value_cache.ptr<f32>();

      std::memcpy(key_cache_ptr + cache_idx * config_.kv_dim, k_batch_ptr, config_.kv_dim * sizeof(f32));
      std::memcpy(value_cache_ptr + cache_idx * config_.kv_dim, v_batch_ptr, config_.kv_dim * sizeof(f32));
    }
  }

  // 4. Batched Multi-Head Attention with paged cache
#ifdef PHOTON_USE_CUDA
  if (config_.device == DeviceType::CUDA) {
    // Use cache_offsets_gpu provided by llama_model (prepared once for all layers!)
    std::span<f32> output_span(attn_result_batch_.ptr<f32>(), attn_result_batch_.size());
    std::span<const f32> query_span(q_batch_.ptr<f32>(), q_batch_.size());
    std::span<f32> score_span(cached_score_buf_.ptr<f32>(), cached_score_buf_.size());
    std::span<const f32> key_span(key_cache.ptr<f32>(), key_cache.size());
    std::span<const f32> value_span(value_cache.ptr<f32>(), value_cache.size());

    auto mha_result = kernels::cuda::batched_mha_paged_launch(
        positions_gpu,  // Use GPU pointer for kernel
        cache_offsets_gpu,
        batch_size,
        config_.n_heads,
        config_.seq_len,
        config_.kv_dim,
        config_.kv_mul,
        config_.head_size,
        output_span,
        query_span,
        score_span,
        key_span,
        value_span,
        nullptr);

    if (!mha_result) return mha_result;
  } else
#endif
  {
    // CPU fallback: process per-sequence with cache copying
    for (i32 i = 0; i < batch_size; ++i) {
      i32 seq_id = seq_ids_cpu[i];  // Use CPU vector
      i32 seq_pos = positions_cpu[i];  // Use CPU vector

      auto offset_result = cache_manager->get_sequence_offset(seq_id);
      if (!offset_result) {
        return Err<void>(offset_result.error());
      }
      i32 seq_offset = offset_result.value();

      f32* q_batch_ptr = q_batch_.ptr<f32>() + i * config_.dim;
      std::memcpy(q_.ptr<f32>(), q_batch_ptr, config_.dim * sizeof(f32));

      i32 tokens_to_copy = seq_pos + 1;
      usize cache_bytes = tokens_to_copy * config_.kv_dim * sizeof(f32);
      f32* key_src = key_cache.ptr<f32>() + seq_offset * config_.kv_dim;
      f32* value_src = value_cache.ptr<f32>() + seq_offset * config_.kv_dim;

      std::memcpy(temp_key_cache_.data(), key_src, cache_bytes);
      std::memcpy(temp_value_cache_.data(), value_src, cache_bytes);

      auto mha_result = mha_.forward(q_, temp_key_cache_, temp_value_cache_, attn_result_, seq_pos);
      if (!mha_result) return mha_result;

      f32* attn_result_ptr = attn_result_batch_.ptr<f32>() + i * config_.dim;
      std::memcpy(attn_result_ptr, attn_result_.ptr<f32>(), config_.dim * sizeof(f32));
    }
  }

  // 5. Output projection (batched)
  auto wo_result = wo_.forward(attn_result_batch_, wo_out_batch_);
  if (!wo_result) return wo_result;

  // 6. Residual connection: x = x + wo_out (batched)
  auto add1_result = add_.forward(x_batch, wo_out_batch_, x_batch);
  if (!add1_result) return add1_result;

  // ========================================================================
  // Feed-Forward Block
  // ========================================================================

  // 7. Pre-FFN RMSNorm (batched)
  auto ffn_norm_result = ffn_norm_.forward(x_batch, ffn_out_batch_);
  if (!ffn_norm_result) return ffn_norm_result;

  // 8. FFN: w2(swiglu(w1(h), w3(h))) - all batched
  auto w1_result = w1_.forward(ffn_out_batch_, w1_out_batch_);
  if (!w1_result) return w1_result;

  auto w3_result = w3_.forward(ffn_out_batch_, w3_out_batch_);
  if (!w3_result) return w3_result;

  auto swiglu_result = swiglu_.forward(w1_out_batch_, w3_out_batch_, swiglu_out_batch_);
  if (!swiglu_result) return swiglu_result;

  auto w2_result = w2_.forward(swiglu_out_batch_, w2_out_batch_);
  if (!w2_result) return w2_result;

  // 9. Residual connection: x = x + w2_out (batched)
  auto add2_result = add_.forward(x_batch, w2_out_batch_, x_batch);
  if (!add2_result) return add2_result;

  return Ok();
}

// ============================================================================
// Weight Quantization
// ============================================================================

Result<void> TransformerBlock::quantize_weights(i32 group_size) {
  // Quantize all MatMul weights in this block
  std::vector<std::pair<const char*, MatMulOp*>> matmuls = {
      {"wq", &wq_},
      {"wk", &wk_},
      {"wv", &wv_},
      {"wo", &wo_},
      {"w1", &w1_},
      {"w2", &w2_},
      {"w3", &w3_}
  };

  for (const auto& [name, matmul] : matmuls) {
    auto result = matmul->quantize_weight(group_size);
    if (!result) {
      return Err<void>(result.error().code(),
                      std::string("Failed to quantize ") + name + " in layer " +
                      std::to_string(layer_idx_) + ": " + result.error().message());
    }
  }

  return Ok();
}

}  // namespace photon::model
