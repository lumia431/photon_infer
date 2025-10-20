/**
 * @file transformer_block.cpp
 * @brief Transformer block implementation
 * @version 0.1.0
 */

#include "photon/model/transformer_block.hpp"

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
      swiglu_(config.hidden_dim) {}

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
  auto device = DeviceType::CPU;
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

Result<void> TransformerBlock::set_wq(Tensor weight) {
  auto result = wq_.set_weight(std::move(weight));
  if (!result) return result;
  return wq_.init();
}

Result<void> TransformerBlock::set_wk(Tensor weight) {
  auto result = wk_.set_weight(std::move(weight));
  if (!result) return result;
  return wk_.init();
}

Result<void> TransformerBlock::set_wv(Tensor weight) {
  auto result = wv_.set_weight(std::move(weight));
  if (!result) return result;
  return wv_.init();
}

Result<void> TransformerBlock::set_wo(Tensor weight) {
  auto result = wo_.set_weight(std::move(weight));
  if (!result) return result;
  return wo_.init();
}

Result<void> TransformerBlock::set_w1(Tensor weight) {
  auto result = w1_.set_weight(std::move(weight));
  if (!result) return result;
  return w1_.init();
}

Result<void> TransformerBlock::set_w2(Tensor weight) {
  auto result = w2_.set_weight(std::move(weight));
  if (!result) return result;
  return w2_.init();
}

Result<void> TransformerBlock::set_w3(Tensor weight) {
  auto result = w3_.set_weight(std::move(weight));
  if (!result) return result;
  return w3_.init();
}

Result<void> TransformerBlock::set_attn_norm(Tensor weight) {
  auto result = attn_norm_.set_weight(std::move(weight));
  if (!result) return result;
  return attn_norm_.init();
}

Result<void> TransformerBlock::set_ffn_norm(Tensor weight) {
  auto result = ffn_norm_.set_weight(std::move(weight));
  if (!result) return result;
  return ffn_norm_.init();
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
  {
    f32* k_ptr = k_.ptr<f32>();
    f32* key_cache_ptr = key_cache.ptr<f32>();
    for (i32 i = 0; i < config_.kv_dim; ++i) {
      key_cache_ptr[pos * config_.kv_dim + i] = k_ptr[i];
    }
  }

  // Copy v_ into value_cache[pos, :]
  {
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

}  // namespace photon::model
