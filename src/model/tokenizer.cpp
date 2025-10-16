#include "photon/model/tokenizer.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>

namespace photon::model {

// Compile-time base64 lookup table
constexpr auto TikTokenizer::build_base64_table() noexcept -> std::array<i32, 256> {
  std::array<i32, 256> table{};

  // Initialize all to -1
  for (auto& val : table) {
    val = -1;
  }

  // Build lookup table
  constexpr std::string_view base64_chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  for (usize i = 0; i < base64_chars.size(); ++i) {
    table[static_cast<u8>(base64_chars[i])] = static_cast<i32>(i);
  }

  return table;
}

auto TikTokenizer::base64_decode(std::string_view encoded) -> std::string {
  static constexpr auto lookup_table = build_base64_table();

  std::string decoded;
  decoded.reserve(encoded.size() * 3 / 4);

  i32 val = 0;
  i32 valb = -8;

  for (u8 c : encoded) {
    const i32 lookup = lookup_table[c];
    if (lookup == -1) break;

    val = (val << 6) + lookup;
    valb += 6;

    if (valb >= 0) {
      decoded.push_back(static_cast<char>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }

  return decoded;
}

void TikTokenizer::add_special_tokens() {
  // Add Llama3 special tokens (128000-128255)
  for (i32 i = kSpecialTokenStart; i <= kSpecialTokenEnd; ++i) {
    if (id_to_token_.find(i) == id_to_token_.end()) {
      std::string special_token =
          "<|reserved_special_token_" + std::to_string(i - kSpecialTokenStart) + "|>";
      id_to_token_[i] = std::move(special_token);
    }
  }

  // Common Llama3 special tokens
  if (id_to_token_.find(128000) == id_to_token_.end()) {
    id_to_token_[128000] = "<|begin_of_text|>";
  }
  if (id_to_token_.find(128001) == id_to_token_.end()) {
    id_to_token_[128001] = "<|end_of_text|>";
  }
}

Result<TikTokenizer> TikTokenizer::load(const std::filesystem::path& vocab_path) {

  if (!std::filesystem::exists(vocab_path)) {
    return Err<TikTokenizer>(ErrorCode::FileNotFound,
      "Tokenizer file not found: " + vocab_path.string());
  }

  std::ifstream file(vocab_path);
  if (!file.is_open()) {
    return Err<TikTokenizer>(ErrorCode::IOError,
      "Failed to open tokenizer file: " + vocab_path.string());
  }

  TikTokenizer tokenizer;
  std::string line;
  i32 max_token_id = 0;

  while (std::getline(file, line)) {
    const usize space_pos = line.find(' ');
    if (space_pos == std::string::npos) continue;

    std::string_view base64_token = std::string_view(line).substr(0, space_pos);
    const i32 token_id = std::stoi(line.substr(space_pos + 1));

    // Decode token
    std::string token = base64_decode(base64_token);

    // Insert into vocab first, then use the inserted key for id_to_token
    auto [it, inserted] = tokenizer.vocab_.insert({token, token_id});
    if (inserted) {
      tokenizer.id_to_token_[token_id] = it->first;
    }

    max_token_id = std::max(max_token_id, token_id);
  }

  if (tokenizer.vocab_.empty()) {
    return Err<TikTokenizer>(ErrorCode::InvalidArgument,
      "Empty vocabulary loaded from: " + vocab_path.string());
  }

  // Add special tokens
  tokenizer.add_special_tokens();

  return Ok(std::move(tokenizer));
}

auto TikTokenizer::encode(std::string_view text) const -> std::vector<i32> {
  std::vector<i32> tokens;
  tokens.reserve(text.size() / 4); // Heuristic: ~4 chars per token

  usize i = 0;
  while (i < text.length()) {
    // Try longest match first (greedy tokenization)
    const usize max_len = std::min(text.length() - i, usize{16});
    bool found = false;

    for (usize len = max_len; len > 0; --len) {
      const std::string_view substr = text.substr(i, len);

      // Use temporary string for lookup (map requires std::string key)
      const std::string key{substr};
      if (auto it = vocab_.find(key); it != vocab_.end()) {
        tokens.push_back(it->second);
        i += len;
        found = true;
        break;
      }
    }

    if (!found) {
      // Fallback: encode byte-by-byte
      const std::string byte_str{text[i]};
      if (auto it = vocab_.find(byte_str); it != vocab_.end()) {
        tokens.push_back(it->second);
      }
      // else: skip unknown byte (or could use a special UNK token)
      ++i;
    }
  }

  return tokens;
}

auto TikTokenizer::decode(const std::vector<i32>& tokens) const -> std::string {
  std::string text;
  text.reserve(tokens.size() * 4); // Heuristic

  for (i32 token : tokens) {
    if (auto it = id_to_token_.find(token); it != id_to_token_.end()) {
      text += it->second;
    }
  }

  return text;
}

auto TikTokenizer::decode_token(i32 token_id) const -> std::string_view {
  if (auto it = id_to_token_.find(token_id); it != id_to_token_.end()) {
    return it->second;
  }
  return "";
}

} // namespace photon::model
