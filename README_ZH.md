# PhotonInfer

<div align="center">

**高性能 LLM 推理引擎，支持 vLLM 风格的连续批处理**

[English](README_EN.md) | [中文](README_ZH.md) | [在线演示](https://photoninfer.xyz/)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C++-20-orange.svg)](https://en.cppreference.com/w/cpp/20)

</div>

---

## 🚀 性能亮点

PhotonInfer 为大语言模型推理提供**生产级性能**，采用先进的批处理技术。**支持 Llama-3.2 和 Qwen3 模型**。

### 单次推理性能

| 模型 | PhotonInfer | llama.cpp | 加速比 |
|------|-------------|-----------|--------|
| Llama 3.2 1B | 185 tok/s | 252 tok/s | 0.73× (llama.cpp 更快) |

### 批量推理吞吐量

| 批量大小 | PhotonInfer | llama.cpp | 加速比 |
|---------|-------------|-----------|--------|
| 4       | 410 tok/s   | 252 tok/s | **1.63×** |
| 8       | 720 tok/s   | 255 tok/s | **2.82×** |
| 16      | 787 tok/s   | 253 tok/s | **3.07×** |

**测试环境**: NVIDIA A100, Llama 3.2 1B, Q8/INT8 量化

## ✨ 核心特性

### 🎯 **vLLM 风格连续批处理**
- **Token 级动态调度**：在生成过程中随时添加新请求，无需等待批次完成
- **两阶段调度器**：无缝继续运行中的请求，同时接纳新请求
- **请求状态跟踪**：精确的 `num_computed_tokens` 管理，实现高效恢复
- **生产就绪**：适合高并发推理服务，具备实时响应能力

### ⚡ **GPU 优化内核**
- **批量分页注意力机制**：块级 KV cache 管理，高效内存利用
- **向量化内存访问**：使用 `float4` 加载，带宽效率提升 2-4 倍
- **算子融合**：零拷贝 GPU 采样、批量 RoPE、融合归一化
- **INT8 量化**：分组量化，支持 cuBLASLt INT8 GEMM
- **优化的 Softmax**：使用 CUB BlockReduce 实现数值稳定的注意力计算

### 🏗️ **现代 C++20 架构**
- **类型安全的错误处理**：借鉴 Rust 的 `Result<T, E>` 类型，显式错误传播
- **零拷贝设计**：广泛使用 `std::span` 和移动语义
- **设备无关**：CPU 和 CUDA 后端的统一接口
- **Concepts & Ranges**：编译期约束和富有表现力的类型安全

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     连续批处理引擎                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              两阶段调度器                           │   │
│  │  • RUNNING 请求（继续生成）                         │   │
│  │  • WAITING 请求（填充剩余容量）                     │   │
│  │  • Token 级抢占支持                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Transformer 层（批量）                    │   │
│  │  • 批量 RMSNorm（融合）                             │   │
│  │  • INT8 量化 MatMul（cuBLASLt）                     │   │
│  │  • 批量 RoPE（融合）                                │   │
│  │  • 分页多头注意力                                   │   │
│  │    - 向量化 K/V cache 访问（float4）                │   │
│  │    - 优化的 softmax（CUB reduce）                   │   │
│  │    - 长序列分区注意力                               │   │
│  │  • SwiGLU FFN                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           GPU 采样（零拷贝）                        │   │
│  │  • 批量 temperature 缩放                            │   │
│  │  • Top-p/top-k 过滤                                 │   │
│  │  • GPU 上的分类采样                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- **编译器**：GCC 12+（需要 C++20 支持）
- **CMake**：3.20+
- **CUDA Toolkit**：12.0+（测试于 12.5）
- **GPU**：NVIDIA GPU，计算能力 7.0+

### 编译

#### 方式 1：从源码编译

```bash
# 克隆仓库
cd photon_infer

# 配置 CUDA 支持
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPHOTON_BUILD_CUDA=ON ..

# 编译
cmake --build . -j$(nproc)
```

#### 方式 2：使用 Docker（推荐）

```bash
# 拉取预构建的 Docker 镜像
docker pull lumia431/photon_infer:latest

# 使用 GPU 支持运行容器
docker run --rm --gpus all -p 5728:5728 -e PORT=5728 lumia431/photon_infer:latest
```

Web 界面将在 `http://localhost:5728` 可用

## 🔬 技术细节

### INT8 量化
- **分组量化**：可配置组大小（32、64、128）
- **cuBLASLt 集成**：硬件加速的 INT8 GEMM
- **精度损失极小**：在 Llama 模型上困惑度下降 < 1%

### 分页注意力
- **块级 KV cache**：高效内存分配，无碎片化
- **动态序列管理**：每个序列的 cache 偏移量，灵活调度
- **批量 cache 操作**：单个内核处理多序列 K/V 写入

### 连续批处理调度器
- **两阶段调度**：
  1. **阶段 1**：继续所有 RUNNING 请求（无中断）
  2. **阶段 2**：接纳 WAITING 请求填充剩余容量
- **请求状态**：WAITING → RUNNING → FINISHED（支持 PREEMPTED）
- **Token 级粒度**：通过 `num_computed_tokens` 跟踪实现精确恢复

## 🛣️ 发展路线

- [x] **核心基础设施**：Tensor、算子、内存管理
- [x] **LLaMA 模型**：完整 transformer 实现，包含 CPU/GPU 内核
- [x] **INT8 量化**：分组量化 + cuBLASLt
- [x] **分页注意力**：块级 KV cache 管理
- [x] **连续批处理**：vLLM 风格动态请求调度
- [ ] **Flash Attention 2**：针对长序列的 IO 感知注意力
- [ ] **多 GPU 支持**：大模型张量并行
- [ ] **FP16/BF16 混合精度**：现代 GPU 上的更高吞吐量
- [ ] **推测解码**：通过草稿模型实现多 token 生成

## 📖 文档

- [连续批处理设计](docs/continuous_batching.md)
- [性能优化指南](docs/performance.md)
- [API 参考](docs/api.md)

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](docs/contributing.md) 了解指南。

## 📝 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- 架构灵感来自 [vLLM](https://github.com/vllm-project/vllm)
- 内核优化参考 [llama.cpp](https://github.com/ggerganov/llama.cpp)
- 错误处理设计借鉴 Rust 的 `Result<T, E>`

---

**Built with ❤️ for high-performance LLM inference**
