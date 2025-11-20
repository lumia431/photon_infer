// Copyright (c) 2025 PhotonInfer Team. All rights reserved.
//
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mutex>

#include "httplib.h"
#include "json.hpp"

#include "photon/core/tensor.hpp"
#include "photon/io/checkpoint.hpp"
#include "photon/arch/llama_model.hpp"
#include "photon/io/tokenizer.hpp"
#include "photon/runtime/kv_cache_manager.hpp"

using json = nlohmann::json;
using namespace photon;
using namespace photon::model;

// Configuration
struct ServerConfig {
    int port = 5728;
    std::string static_dir = "/photon_infer/web/static";
    std::string checkpoint_path = "./models/llama-3.2-1b-instruct-q8_0.gguf";
    std::string tokenizer_path = "./models/llama-3.2-tokenizer.json";
    int max_seq_len = 2048;
    bool use_https = false;
    std::string cert_file = "";
    std::string key_file = "";
};

// Global model instance
class ModelManager {
private:
    std::unique_ptr<LLaMAModel> model_;
    std::unique_ptr<TikTokenizer> tokenizer_;
    std::string current_model_name_ = "llama-3.2-1b";
    bool is_loaded_ = false;
    i32 vocab_size_ = 0;
    std::mutex mutex_;  // Protect model access
    std::string active_session_id_;  // Track which session is currently loaded in KV cache

public:
    bool load_model(const std::string& checkpoint_path, const std::string& tokenizer_path) {
        try {
            std::cout << "[INFO] Loading tokenizer from: " << tokenizer_path << std::endl;

            // Load tokenizer
            auto tokenizer_result = TikTokenizer::load(tokenizer_path);
            if (!tokenizer_result) {
                std::cerr << "[ERROR] Failed to load tokenizer: " << tokenizer_result.error().message() << std::endl;
                std::cerr << "[ERROR] Note: PhotonInfer requires TikToken format tokenizer, not SentencePiece." << std::endl;
                std::cerr << "[ERROR] Please see web/TOKENIZER_GUIDE.md for format requirements." << std::endl;
                return false;
            }
            tokenizer_ = std::make_unique<TikTokenizer>(std::move(tokenizer_result.value()));
            std::cout << "[INFO] Tokenizer loaded, vocab size: " << tokenizer_->vocab_size() << std::endl;

            // Load checkpoint
            std::cout << "[INFO] Loading checkpoint from: " << checkpoint_path << std::endl;
            auto loader_result = CheckpointLoader::open(checkpoint_path);
            if (!loader_result) {
                std::cerr << "[ERROR] Failed to load checkpoint: " << loader_result.error().message() << std::endl;
                return false;
            }
            auto loader = std::move(loader_result.value());

            const auto& header = loader->header();
            std::cout << "[INFO] Model info - Layers: " << header.n_layers
                      << ", Dim: " << header.dim << std::endl;

            // Create config
            TransformerConfig config;
            config.dim = header.dim;
            config.hidden_dim = header.hidden_dim;
            config.n_layers = header.n_layers;
            config.n_heads = header.n_heads;
            config.n_kv_heads = header.n_kv_heads;
            config.vocab_size = header.vocab_size;
            config.seq_len = header.seq_len;
            config.head_size = header.dim / header.n_heads;
            config.norm_eps = 1e-5f;
            config.device = DeviceType::CUDA;
            config.compute_derived();

            vocab_size_ = config.vocab_size;

            // Create model
            std::cout << "[INFO] Creating model..." << std::endl;
            model_ = std::make_unique<LLaMAModel>(config);

            // Load weights
            std::cout << "[INFO] Loading model weights..." << std::endl;
            auto load_weights_result = loader->load_weights(*model_);
            if (!load_weights_result) {
                std::cerr << "[ERROR] Failed to load weights: " << load_weights_result.error().message() << std::endl;
                return false;
            }

            // Initialize model
            std::cout << "[INFO] Initializing model..." << std::endl;
            auto init_result = model_->init();
            if (!init_result) {
                std::cerr << "[ERROR] Failed to initialize model: " << init_result.error().message() << std::endl;
                return false;
            }

            // Quantize weights (INT8)
            std::cout << "[INFO] Quantizing model weights (INT8)..." << std::endl;
            auto quant_result = model_->quantize_weights(128);
            if (!quant_result) {
                std::cerr << "[ERROR] Failed to quantize model: " << quant_result.error().message() << std::endl;
                return false;
            }

            is_loaded_ = true;
            std::cout << "[INFO] Model loaded successfully!" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception during model loading: " << e.what() << std::endl;
            return false;
        }
    }

    struct SessionState {
        i32 current_pos = 0;
        std::vector<i32> all_tokens;
        Tensor logits;
        std::chrono::steady_clock::time_point last_access;
    };

    std::map<std::string, SessionState> sessions_;

    std::string generate(const std::string& session_id, const std::string& prompt,
                        int max_tokens, float temperature, float /*top_p*/) {
        std::lock_guard<std::mutex> lock(mutex_);  // Thread-safe access

        if (!is_loaded_ || !model_ || !tokenizer_) {
            return "[ERROR] Model not loaded";
        }

        try {
            // Get or create session state
            auto& state = sessions_[session_id];

            // Initialize logits if needed
            if (state.logits.size() == 0) {
                auto logits_result = Tensor::create({vocab_size_}, DataType::Float32, DeviceType::CPU);
                if (!logits_result) {
                    return "[ERROR] Failed to create logits tensor";
                }
                state.logits = std::move(logits_result.value());
            }

            state.last_access = std::chrono::steady_clock::now();

            // Check if we need to rebuild KV cache due to session switch
            bool need_rebuild = (active_session_id_ != session_id);
            if (need_rebuild && !state.all_tokens.empty()) {
                std::cout << "[INFO] Session switch detected, rebuilding KV cache" << std::endl;

                state.current_pos = 0;
                for (size_t i = 0; i < state.all_tokens.size(); ++i) {
                    auto forward_result = model_->forward(state.all_tokens[i], state.current_pos, state.logits);
                    if (!forward_result) {
                        return "[ERROR] Failed to rebuild KV cache: " + forward_result.error().message();
                    }
                    state.current_pos++;
                }
            }
            active_session_id_ = session_id;

            // Tokenize new prompt
            auto new_tokens = tokenizer_->encode(prompt);
            if (new_tokens.empty()) {
                return "[ERROR] Failed to tokenize input";
            }

            // Add BOS token only for first turn
            if (state.all_tokens.empty()) {
                state.all_tokens.push_back(tokenizer_->bos_id());
            }

            i32 start_pos = state.current_pos;
            state.all_tokens.insert(state.all_tokens.end(), new_tokens.begin(), new_tokens.end());

            // Process prompt tokens
            for (size_t i = start_pos; i < state.all_tokens.size(); ++i) {
                auto forward_result = model_->forward(state.all_tokens[i], state.current_pos, state.logits);
                if (!forward_result) {
                    return "[ERROR] Forward pass failed: " + forward_result.error().message();
                }
                state.current_pos++;
            }

            // Generate new tokens
            std::vector<i32> generated_tokens;
            i32 tokens_generated = 0;
            i32 repeat_count = 0;
            i32 last_token = -1;

            while (tokens_generated < max_tokens) {
                i32 next_token = sample_token(state.logits, temperature);

                // Check for EOS token
                if (next_token == tokenizer_->eos_id() || next_token == 128001) {
                    break;
                }

                // Detect infinite loops of repeated special tokens
                if (next_token == last_token && next_token >= 128000) {
                    repeat_count++;
                    if (repeat_count > 5) {
                        break;
                    }
                } else {
                    repeat_count = 0;
                    last_token = next_token;
                }

                generated_tokens.push_back(next_token);
                state.all_tokens.push_back(next_token);
                tokens_generated++;

                auto forward_result = model_->forward(next_token, state.current_pos, state.logits);
                if (!forward_result) {
                    std::cerr << "[ERROR] Forward pass failed: " << forward_result.error().message() << std::endl;
                    break;
                }
                state.current_pos++;
            }

            // Decode generated tokens
            std::string result;
            for (i32 token : generated_tokens) {
                result += tokenizer_->decode_token(token);
            }

            std::cout << "[INFO] Generated " << tokens_generated << " tokens" << std::endl;

            // Strip special tokens from output
            result = strip_special_tokens(result);

            // Force CUDA synchronization and memory cleanup
#ifdef PHOTON_USE_CUDA
            cudaDeviceSynchronize();
#endif

            return result;

        } catch (const std::exception& e) {
            // Force CUDA cleanup on error
#ifdef PHOTON_USE_CUDA
            cudaDeviceSynchronize();
#endif
            return std::string("[ERROR] Exception during generation: ") + e.what();
        }
    }

    void clear_session(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.erase(session_id);
        if (active_session_id_ == session_id) {
            active_session_id_.clear();
        }
        std::cout << "[INFO] Cleared session: " << session_id << std::endl;
    }

    void cleanup_old_sessions(int timeout_minutes = 30) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();

        for (auto it = sessions_.begin(); it != sessions_.end(); ) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.last_access).count();
            if (elapsed > timeout_minutes) {
                std::cout << "[INFO] Removing stale session: " << it->first << std::endl;
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool is_loaded() const { return is_loaded_; }
    std::string current_model() const { return current_model_name_; }

private:
    i32 sample_token(const Tensor& logits, float temperature) {
        const f32* logits_ptr = logits.ptr<f32>();

        std::vector<f32> probs(vocab_size_);
        f32 max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size_);

        f32 sum_exp = 0.0f;
        for (i32 i = 0; i < vocab_size_; ++i) {
            probs[i] = std::exp((logits_ptr[i] - max_logit) / temperature);
            sum_exp += probs[i];
        }

        for (i32 i = 0; i < vocab_size_; ++i) {
            probs[i] /= sum_exp;
        }

        f32 rand_val = static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX);
        f32 cumulative = 0.0f;
        for (i32 i = 0; i < vocab_size_; ++i) {
            cumulative += probs[i];
            if (rand_val <= cumulative) {
                return i;
            }
        }

        return 0;
    }

    std::string strip_special_tokens(const std::string& text) {
        std::string result = text;

        // Remove all <|...|> patterns (Llama special tokens)
        size_t pos = 0;
        while ((pos = result.find("<|", pos)) != std::string::npos) {
            size_t end = result.find("|>", pos);
            if (end != std::string::npos) {
                // Found a complete <|...|> pattern, remove it
                result.erase(pos, end - pos + 2);
                // Don't increment pos, check same position again
            } else {
                // No matching |>, move past this <|
                pos += 2;
            }
        }

        return result;
    }
};

// CORS middleware
void enable_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

int main(int argc, char** argv) {
    ServerConfig config;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if ((arg == "--checkpoint" || arg == "--model") && i + 1 < argc) {
            config.checkpoint_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            config.tokenizer_path = argv[++i];
        } else if (arg == "--static-dir" && i + 1 < argc) {
            config.static_dir = argv[++i];
        } else if (arg == "--https") {
            config.use_https = true;
        } else if (arg == "--cert" && i + 1 < argc) {
            config.cert_file = argv[++i];
        } else if (arg == "--key" && i + 1 < argc) {
            config.key_file = argv[++i];
        }
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "========================================" << std::endl;
    std::cout << "  PhotonInfer Web Server" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[INFO] Port: " << config.port << std::endl;
    std::cout << "[INFO] HTTPS: " << (config.use_https ? "enabled" : "disabled") << std::endl;
    if (config.use_https) {
        std::cout << "[INFO] Cert file: " << config.cert_file << std::endl;
        std::cout << "[INFO] Key file: " << config.key_file << std::endl;
    }
    std::cout << "[INFO] Static directory: " << config.static_dir << std::endl;
    std::cout << "[INFO] Checkpoint: " << config.checkpoint_path << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize model manager
    ModelManager model_manager;

    std::cout << "[INFO] Loading model... This may take a few moments." << std::endl;
    if (!model_manager.load_model(config.checkpoint_path, config.tokenizer_path)) {
        std::cerr << "[ERROR] Failed to load model. Server will start but inference will not work." << std::endl;
    }

    // Create HTTP/HTTPS server
    std::unique_ptr<httplib::Server> svr_ptr;

    if (config.use_https) {
        if (config.cert_file.empty() || config.key_file.empty()) {
            std::cerr << "[ERROR] HTTPS enabled but cert/key files not specified" << std::endl;
            std::cerr << "[ERROR] Use --cert and --key options to specify certificate files" << std::endl;
            return 1;
        }
        svr_ptr = std::make_unique<httplib::SSLServer>(config.cert_file.c_str(), config.key_file.c_str());
    } else {
        svr_ptr = std::make_unique<httplib::Server>();
    }

    auto& svr = *svr_ptr;

    // CORS preflight handler
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        enable_cors(res);
        res.status = 200;
    });

    // Serve static files
    svr.set_mount_point("/", config.static_dir);

    // API: Health check
    svr.Get("/api/health", [&](const httplib::Request&, httplib::Response& res) {
        enable_cors(res);
        json response = {
            {"status", "ok"},
            {"model_loaded", model_manager.is_loaded()},
            {"current_model", model_manager.current_model()}
        };
        res.set_content(response.dump(), "application/json");
    });

    // API: Chat endpoint
    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        try {
            auto request_json = json::parse(req.body);

            std::string message = request_json.value("message", "");
            std::string session_id = request_json.value("session_id", "default");
            int max_tokens = request_json.value("max_tokens", 128);
            float temperature = request_json.value("temperature", 0.7f);
            float top_p = request_json.value("top_p", 0.9f);

            if (message.empty()) {
                json error_response = {
                    {"error", "Message is required"}
                };
                res.set_content(error_response.dump(), "application/json");
                res.status = 400;
                return;
            }

            std::cout << "[INFO] Received chat request (session: " << session_id << "): "
                      << message.substr(0, 50) << (message.length() > 50 ? "..." : "") << std::endl;

            std::string prompt = message;

            auto start_time = std::chrono::high_resolution_clock::now();

            // Generate response
            std::string response_text = model_manager.generate(session_id, prompt, max_tokens, temperature, top_p);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            std::cout << "[INFO] Generation completed in " << duration << "ms" << std::endl;

            json response = {
                {"response", response_text},
                {"tokens", max_tokens},
                {"duration_ms", duration}
            };

            res.set_content(response.dump(), "application/json");

        } catch (const json::exception& e) {
            json error_response = {
                {"error", std::string("JSON error: ") + e.what()}
            };
            res.set_content(error_response.dump(), "application/json");
            res.status = 400;
        } catch (const std::exception& e) {
            json error_response = {
                {"error", std::string("Server error: ") + e.what()}
            };
            res.set_content(error_response.dump(), "application/json");
            res.status = 500;
        }
    });

    // API: Clear session endpoint
    svr.Post("/api/clear_session", [&](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        try {
            auto request_json = json::parse(req.body);
            std::string session_id = request_json.value("session_id", "default");

            model_manager.clear_session(session_id);

            json response = {
                {"success", true},
                {"session_id", session_id}
            };
            res.set_content(response.dump(), "application/json");

        } catch (const json::exception& e) {
            json error_response = {
                {"success", false},
                {"error", std::string("JSON error: ") + e.what()}
            };
            res.set_content(error_response.dump(), "application/json");
        }
    });

    // API: Switch model endpoint
    svr.Post("/api/switch_model", [&](const httplib::Request& req, httplib::Response& res) {
        enable_cors(res);

        try {
            auto request_json = json::parse(req.body);
            std::string model = request_json.value("model", "");

            // Currently only llama-3.2-1b is supported
            if (model != "llama-3.2-1b") {
                json response = {
                    {"success", false},
                    {"error", "Only llama-3.2-1b is currently supported"}
                };
                res.set_content(response.dump(), "application/json");
                return;
            }

            json response = {
                {"success", true},
                {"model", model}
            };
            res.set_content(response.dump(), "application/json");

        } catch (const json::exception& e) {
            json error_response = {
                {"success", false},
                {"error", std::string("JSON error: ") + e.what()}
            };
            res.set_content(error_response.dump(), "application/json");
        }
    });

    // Start server
    std::string protocol = config.use_https ? "https" : "http";
    std::cout << "[INFO] Starting server on " << protocol << "://localhost:" << config.port << std::endl;
    std::cout << "[INFO] Open " << protocol << "://localhost:" << config.port << " in your browser" << std::endl;
    std::cout << "========================================" << std::endl;

    if (!svr.listen("0.0.0.0", config.port)) {
        std::string protocol_name = config.use_https ? "HTTPS" : "HTTP";
        std::cerr << "[ERROR] Failed to start " << protocol_name << " server on port " << config.port << std::endl;
        return 1;
    }

    return 0;
}
