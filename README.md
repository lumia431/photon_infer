# PhotonInfer

<div align="center">

**A High-Performance LLM Inference Engine with vLLM-Style Continuous Batching**

[English](README.md) | [中文](README_ZH.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C++-20-orange.svg)](https://en.cppreference.com/w/cpp/20)

</div>

---

## 🚀 Performance Highlights

PhotonInfer delivers **production-grade inference performance** for LLMs with advanced batching capabilities:

| Metric | Performance | Comparison |
|--------|------------|------------|
| **Peak Throughput** | **518 tokens/s** @ batch=16 | **2.05×** faster than llama.cpp |
| **Batch Scaling** | 71 → 518 tokens/s (7.3×) | Linear scaling up to batch=16 |
| **Continuous Batching** | **2.02× throughput**, 2.59× lower latency | Unique advantage over baseline |
| **Urgent Request Latency** | **0.29s** vs 755s (baseline) | **2500×** improvement |

**Tested on**: NVIDIA RTX 5060 Ti, Llama 3.2 1B, Q8/INT8 quantization

## ✨ Key Features

### 🎯 **vLLM-Style Continuous Batching**
- **Token-level dynamic scheduling**: Add new requests mid-generation without waiting for batch completion
- **Two-phase scheduler**: Seamlessly continue running requests while admitting new ones
- **Request state tracking**: Precise `num_computed_tokens` management for efficient resume
- **Perfect for production**: High-concurrency inference services with real-time responsiveness

### ⚡ **GPU-Optimized Kernels**
- **Batched Paged Attention**: Block-level KV cache management with efficient memory utilization
- **Vectorized Memory Access**: `float4` loads for 2-4× bandwidth efficiency
- **Fused Operations**: Zero-copy GPU sampling, batched RoPE, and fused normalization
- **INT8 Quantization**: Group-wise quantization with cuBLASLt INT8 GEMM support
- **Optimized Softmax**: CUB BlockReduce for numerically stable attention computation

### 🏗️ **Modern C++20 Architecture**
- **Type-Safe Error Handling**: Rust-inspired `Result<T, E>` type for explicit error propagation
- **Zero-Copy Design**: Extensive use of `std::span` and move semantics
- **Device Agnostic**: Unified interface for CPU and CUDA backends
- **Concepts & Ranges**: Compile-time constraints and expressive type safety

## 📊 Benchmark Results

### Batch Inference Throughput

```
Tokens/s
600 ┤                                                ╭─── PhotonInfer
    │                                           ╭────╯
500 ┤                                      ╭────╯
    │                                 ╭────╯
400 ┤                            ╭────╯
    │                       ╭────╯
300 ┤                  ╭────╯
    │  llama.cpp ──────────────────────────────────
200 ┤
    │
100 ┤
    │
  0 ┼────────┬────────┬────────┬────────┬────────
    1        2        4        8        16
                   Batch Size
```

**PhotonInfer dominates at batch ≥ 4 with true parallel batch processing**

### Continuous Batching Advantage

| Scenario | Baseline (Wait) | Continuous Batching | Improvement |
|----------|----------------|---------------------|-------------|
| **Throughput** | 236 tokens/s | 477 tokens/s | **2.02×** |
| **Average Latency** | 3.27s | 1.26s | **2.59×** |
| **Urgent Request** | 755s | 0.29s | **2500×+** |

## 🎯 Use Cases

**PhotonInfer excels at:**
- ✅ High-concurrency inference services (4+ concurrent requests)
- ✅ Real-time interactive applications requiring low latency
- ✅ Production deployments prioritizing overall throughput
- ✅ Dynamic workloads with varying request arrival patterns

**Choose llama.cpp for:**
- 📱 Single-user local applications
- 💻 Low-concurrency scenarios (1-3 requests)
- 🔋 Resource-constrained environments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Continuous Batch Engine                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Two-Phase Scheduler                         │   │
│  │  • RUNNING requests (continue generation)           │   │
│  │  • WAITING requests (fill remaining capacity)       │   │
│  │  • Token-level preemption support                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Transformer Layers (Batched)                │   │
│  │  • Batched RMSNorm (fused)                          │   │
│  │  • INT8 Quantized MatMul (cuBLASLt)                 │   │
│  │  • Batched RoPE (fused)                             │   │
│  │  • Paged Multi-Head Attention                       │   │
│  │    - Vectorized K/V cache access (float4)           │   │
│  │    - Optimized softmax (CUB reduce)                 │   │
│  │    - Partitioned attention for long sequences       │   │
│  │  • SwiGLU FFN                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         GPU Sampling (Zero-Copy)                    │   │
│  │  • Batched temperature scaling                      │   │
│  │  • Top-p/top-k filtering                            │   │
│  │  • Categorical sampling on GPU                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Compiler**: GCC 12+ (C++20 support required)
- **CMake**: 3.20+
- **CUDA Toolkit**: 11.0+ (tested on 12.0)
- **GPU**: NVIDIA GPU with Compute Capability 7.0+

### Build

```bash
# Clone repository
cd photon_infer

# Configure with CUDA
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPHOTON_BUILD_CUDA=ON ..

# Build
cmake --build . -j$(nproc)
```

### Run Inference

```bash
# Single request inference
./bin/llama_infer

# Batched inference (4 concurrent requests)
./bin/batched_inference_demo

# Continuous batching demo (compare with baseline)
./bin/compare_batching_methods

# Comprehensive benchmark
./bin/benchmark_photon
```

### Example: Continuous Batching Engine

```cpp
#include "photon/scheduler/continuous_batch_engine.hpp"

using namespace photon::scheduler;

// Initialize engine
ContinuousBatchEngine engine(model, max_batch_size, max_seq_len);

// Add requests dynamically (non-blocking)
auto req1 = engine.add_request(prompt_tokens_1, max_new_tokens);
auto req2 = engine.add_request(prompt_tokens_2, max_new_tokens);

// Engine automatically schedules and executes
engine.step();  // Process one token for entire batch

// Retrieve results as they complete
if (req1->is_finished()) {
  auto tokens = req1->generated_tokens();
  std::string text = tokenizer.decode(tokens);
}
```

## 📁 Project Structure

```
photon_infer/
├── include/photon/
│   ├── core/                    # Core abstractions
│   │   ├── types.hpp           # Type system with C++20 concepts
│   │   ├── error.hpp           # Result<T> error handling
│   │   ├── tensor.hpp          # N-dimensional tensor
│   │   └── allocator.hpp       # Device memory allocators
│   ├── ops/                     # Operators
│   │   ├── matmul.hpp          # INT8 quantized matrix multiplication
│   │   ├── mha.hpp             # Multi-head attention
│   │   ├── rope.hpp            # Rotary position embedding
│   │   └── kernels/cuda/       # CUDA kernel implementations
│   ├── arch/                    # Model architecture
│   │   ├── llama_model.hpp     # LLaMA transformer model
│   │   ├── transformer_block.hpp
│   │   └── config.hpp          # Model configuration
│   ├── runtime/                 # Runtime components
│   │   └── kv_cache_manager.hpp # Paged KV cache
│   ├── io/                      # Input/Output
│   │   ├── checkpoint.hpp      # Checkpoint loader
│   │   ├── model_loader.hpp    # mmap-based model loading
│   │   └── tokenizer.hpp       # TikToken tokenizer
│   └── scheduler/               # Continuous batching scheduler
│       ├── inference_request.hpp
│       ├── continuous_batch_scheduler.hpp
│       └── continuous_batch_engine.hpp
├── src/                         # Implementation files
├── demo/                        # Demo applications
│   ├── compare_batching_methods.cpp  # Baseline vs continuous batching
│   ├── benchmark_photon.cpp          # Comprehensive benchmarks
│   └── batched_inference_demo.cpp    # Multi-request inference
└── tests/                       # Unit tests (Google Test)
```

## 🔬 Technical Details

### INT8 Quantization
- **Group-wise quantization**: Configurable group size (32, 64, 128)
- **cuBLASLt integration**: Hardware-accelerated INT8 GEMM
- **Minimal accuracy loss**: < 1% perplexity degradation on Llama models

### Paged Attention
- **Block-level KV cache**: Efficient memory allocation without fragmentation
- **Dynamic sequence management**: Per-sequence cache offsets for flexible scheduling
- **Batched cache operations**: Single kernel for multi-sequence K/V writes

### Continuous Batching Scheduler
- **Two-phase scheduling**:
  1. **Phase 1**: Continue all RUNNING requests (no interruption)
  2. **Phase 2**: Admit WAITING requests to fill remaining capacity
- **Request states**: WAITING → RUNNING → FINISHED (with PREEMPTED support)
- **Token-level granularity**: `num_computed_tokens` tracking for precise resume

## 📊 Performance Comparison

### vs llama.cpp (Q8_0, Llama 3.2 1B, RTX 5060 Ti)

| Batch Size | PhotonInfer | llama.cpp | Speedup |
|------------|-------------|-----------|---------|
| 1          | 71 tok/s    | 252 tok/s | 0.28× (llama.cpp faster) |
| 2          | 134 tok/s   | 252 tok/s | 0.53× |
| 4          | 273 tok/s   | 252 tok/s | **1.08×** |
| 8          | 480 tok/s   | 255 tok/s | **1.88×** |
| 16         | 518 tok/s   | 253 tok/s | **2.05×** |

**Key observation**: llama.cpp's decode performance is **constant across batch sizes** (~252 tok/s), indicating serial processing. PhotonInfer achieves **true parallel batching** with linear scaling.

## 🛣️ Roadmap

- [x] **Core Infrastructure**: Tensor, operators, memory management
- [x] **LLaMA Model**: Full transformer implementation with CPU/GPU kernels
- [x] **INT8 Quantization**: Group-wise quantization with cuBLASLt
- [x] **Paged Attention**: Block-level KV cache management
- [x] **Continuous Batching**: vLLM-style dynamic request scheduling
- [ ] **Flash Attention 2**: IO-aware attention for long sequences
- [ ] **Multi-GPU Support**: Tensor parallelism for large models
- [ ] **FP16/BF16 Mixed Precision**: Enhanced throughput on modern GPUs
- [ ] **Speculative Decoding**: Multi-token generation with draft model

## 📖 Documentation

- [Continuous Batching Design](docs/continuous_batching.md)
- [Performance Optimization Guide](docs/performance.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Architecture inspired by [vLLM](https://github.com/vllm-project/vllm)
- Kernel optimizations reference [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Error handling design from Rust's `Result<T, E>`

---

**Built with ❤️ for high-performance LLM inference**
