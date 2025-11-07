/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */
#pragma once


#include "photon/core/error.hpp"
#include "photon/core/types.hpp"
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace photon::model {

/**
 * @brief Modern C++20 TikToken tokenizer for Llama3.2 models
 *
 * Features:
 * - Base64 decoding for token strings
 * - Greedy longest-match tokenization
 * - Support for special tokens (BOS, EOS, reserved tokens)
 * - Efficient encoding/decoding using std::map
 *
 * File format: Each line is "base64_token token_id"
 */
class TikTokenizer {
public:
  /**
   * @brief Load tokenizer from vocab file
   * @param vocab_path Path to the tokenizer.model file (base64 format)
   * @return Result containing tokenizer or error
   */
  [[nodiscard]] static Result<TikTokenizer> load(const std::filesystem::path& vocab_path);

  /**
   * @brief Encode text into token IDs
   * @param text Input text to encode
   * @return Vector of token IDs
   */
  [[nodiscard]] auto encode(std::string_view text) const -> std::vector<i32>;

  /**
   * @brief Decode token IDs back to text
   * @param tokens Vector of token IDs
   * @return Decoded text string
   */
  [[nodiscard]] auto decode(const std::vector<i32>& tokens) const -> std::string;

  /**
   * @brief Decode a single token ID to text
   */
  [[nodiscard]] auto decode_token(i32 token_id) const -> std::string_view;

  // Special token accessors
  [[nodiscard]] i32 vocab_size() const noexcept { return static_cast<i32>(id_to_token_.size()); }
  [[nodiscard]] i32 bos_id() const noexcept { return bos_id_; }
  [[nodiscard]] i32 eos_id() const noexcept { return eos_id_; }

  // Llama3.2 special tokens
  static constexpr i32 kDefaultBosId = 128000;
  static constexpr i32 kDefaultEosId = 128001;
  static constexpr i32 kSpecialTokenStart = 128000;
  static constexpr i32 kSpecialTokenEnd = 128255;

private:
  TikTokenizer() = default;

  /**
   * @brief Base64 decode a string (compile-time optimized lookup table)
   */
  [[nodiscard]] static auto base64_decode(std::string_view encoded) -> std::string;

  /**
   * @brief Build base64 lookup table at compile time
   */
  [[nodiscard]] static constexpr auto build_base64_table() noexcept -> std::array<i32, 256>;

  /**
   * @brief Add Llama3 special tokens to vocabulary
   */
  void add_special_tokens();

  std::map<std::string, i32> vocab_;        ///< Token string -> ID
  std::map<i32, std::string> id_to_token_;  ///< ID -> Token string
  i32 bos_id_ = kDefaultBosId;
  i32 eos_id_ = kDefaultEosId;
};

} // namespace photon::model

