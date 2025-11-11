# PhotonInfer

<div align="center">

**A High-Performance LLM Inference Engine with vLLM-Style Continuous Batching**

[English](README.md) | [中文](README_ZH.md) | [Live Demo](https://photoninfer.xyz/)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C++-20-orange.svg)](https://en.cppreference.com/w/cpp/20)

</div>

---

## 🚀 Performance Highlights

PhotonInfer delivers **production-grade inference performance** for LLMs with advanced batching capabilities. **Supports Llama-3.2 and Qwen3 models**.

### Single Request Inference

| Model | PhotonInfer | llama.cpp | Speedup |
|-------|-------------|-----------|---------|
| Llama 3.2 1B | 185 tok/s | 252 tok/s | 0.73× (llama.cpp faster) |

### Batched Inference Throughput

| Batch Size | PhotonInfer | llama.cpp | Speedup |
|------------|-------------|-----------|---------|
| 4          | 410 tok/s   | 252 tok/s | **1.63×** |
| 8          | 720 tok/s   | 255 tok/s | **2.82×** |
| 16         | 787 tok/s   | 253 tok/s | **3.07×** |

**Tested on**: NVIDIA A100, Llama 3.2 1B, Q8/INT8 quantization

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
- **CUDA Toolkit**: 12.0+ (tested on 12.5)
- **GPU**: NVIDIA GPU with Compute Capability 7.0+

### Build

#### Option 1: Build from Source

```bash
# Clone repository
cd photon_infer

# Configure with CUDA
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPHOTON_BUILD_CUDA=ON ..

# Build
cmake --build . -j$(nproc)
```

#### Option 2: Use Docker (Recommended)

```bash
# Pull the pre-built Docker image
docker pull lumia431/photon_infer:latest

# Run the container with GPU support
docker run --rm --gpus all -p 5728:5728 -e PORT=5728 lumia431/photon_infer:latest
```

The web interface will be available at `http://localhost:5728`

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
