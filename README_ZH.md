# PhotonInfer

<div align="center">

**高性能 LLM 推理引擎，支持 vLLM 风格的连续批处理**

[English](README_EN.md) | [中文](README_ZH.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C++-20-orange.svg)](https://en.cppreference.com/w/cpp/20)

</div>

---

## 🚀 性能亮点

PhotonInfer 为大语言模型推理提供**生产级性能**，采用先进的批处理技术：

| 指标 | 性能表现 | 对比优势 |
|------|---------|----------|
| **峰值吞吐量** | **518 tokens/s** @ batch=16 | 比 llama.cpp **快 2.05 倍** |
| **批量扩展性** | 71 → 518 tokens/s (7.3×) | 线性扩展至 batch=16 |
| **连续批处理** | **2.02× 吞吐量**, 2.59× 更低延迟 | 相比基线的独特优势 |
| **紧急请求延迟** | **0.29s** vs 755s (基线) | **2500 倍**改进 |

**测试环境**: NVIDIA RTX 5060 Ti, Llama 3.2 1B, Q8/INT8 量化

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

## 📊 性能测试结果

### 批量推理吞吐量

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
                   批量大小
```

**PhotonInfer 在 batch ≥ 4 时通过真正的并行批处理占据优势**

### 连续批处理优势

| 场景 | 基线（等待） | 连续批处理 | 提升 |
|------|------------|-----------|------|
| **吞吐量** | 236 tokens/s | 477 tokens/s | **2.02×** |
| **平均延迟** | 3.27s | 1.26s | **2.59×** |
| **紧急请求** | 755s | 0.29s | **2500×+** |

## 🎯 适用场景

**PhotonInfer 擅长：**
- ✅ 高并发推理服务（4+ 并发请求）
- ✅ 需要低延迟的实时交互应用
- ✅ 注重整体吞吐量的生产部署
- ✅ 请求到达模式多变的动态工作负载

**选择 llama.cpp 适合：**
- 📱 单用户本地应用
- 💻 低并发场景（1-3 个请求）
- 🔋 资源受限环境

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
- **CUDA Toolkit**：11.0+（测试于 12.0）
- **GPU**：NVIDIA GPU，计算能力 7.0+

### 编译

```bash
# 克隆仓库
cd photon_infer

# 配置 CUDA 支持
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPHOTON_BUILD_CUDA=ON ..

# 编译
cmake --build . -j$(nproc)
```

### 运行推理

```bash
# 单请求推理
./bin/llama_infer

# 批量推理（4 个并发请求）
./bin/batched_inference_demo

# 连续批处理演示（与基线对比）
./bin/compare_batching_methods

# 综合性能测试
./bin/benchmark_photon
```

### 示例：连续批处理引擎

```cpp
#include "photon/scheduler/continuous_batch_engine.hpp"

using namespace photon::scheduler;

// 初始化引擎
ContinuousBatchEngine engine(model, max_batch_size, max_seq_len);

// 动态添加请求（非阻塞）
auto req1 = engine.add_request(prompt_tokens_1, max_new_tokens);
auto req2 = engine.add_request(prompt_tokens_2, max_new_tokens);

// 引擎自动调度和执行
engine.step();  // 为整个批次处理一个 token

// 随时获取完成的结果
if (req1->is_finished()) {
  auto tokens = req1->generated_tokens();
  std::string text = tokenizer.decode(tokens);
}
```

## 📁 项目结构

```
photon_infer/
├── include/photon/
│   ├── core/                    # 核心抽象
│   │   ├── types.hpp           # C++20 concepts 类型系统
│   │   ├── error.hpp           # Result<T> 错误处理
│   │   ├── tensor.hpp          # N 维张量
│   │   └── allocator.hpp       # 设备内存分配器
│   ├── ops/                     # 算子
│   │   ├── matmul.hpp          # INT8 量化矩阵乘法
│   │   ├── mha.hpp             # 多头注意力
│   │   ├── rope.hpp            # 旋转位置编码
│   │   └── kernels/cuda/       # CUDA 内核实现
│   ├── arch/                    # 模型架构
│   │   ├── llama_model.hpp     # LLaMA transformer 模型
│   │   ├── transformer_block.hpp
│   │   └── config.hpp          # 模型配置
│   ├── runtime/                 # 运行时组件
│   │   └── kv_cache_manager.hpp # 分页 KV cache
│   ├── io/                      # 输入输出
│   │   ├── checkpoint.hpp      # Checkpoint 加载器
│   │   ├── model_loader.hpp    # 基于 mmap 的模型加载
│   │   └── tokenizer.hpp       # TikToken 分词器
│   └── scheduler/               # 连续批处理调度器
│       ├── inference_request.hpp
│       ├── continuous_batch_scheduler.hpp
│       └── continuous_batch_engine.hpp
├── src/                         # 实现文件
├── demo/                        # 演示程序
│   ├── compare_batching_methods.cpp  # 基线 vs 连续批处理
│   ├── benchmark_photon.cpp          # 综合性能测试
│   └── batched_inference_demo.cpp    # 多请求推理
└── tests/                       # 单元测试（Google Test）
```

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

## 📊 性能对比

### vs llama.cpp (Q8_0, Llama 3.2 1B, RTX 5060 Ti)

| 批量大小 | PhotonInfer | llama.cpp | 加速比 |
|---------|-------------|-----------|--------|
| 1       | 71 tok/s    | 252 tok/s | 0.28× (llama.cpp 更快) |
| 2       | 134 tok/s   | 252 tok/s | 0.53× |
| 4       | 273 tok/s   | 252 tok/s | **1.08×** |
| 8       | 480 tok/s   | 255 tok/s | **1.88×** |
| 16      | 518 tok/s   | 253 tok/s | **2.05×** |

**关键发现**：llama.cpp 的 decode 性能在所有批量大小下**保持恒定**（~252 tok/s），表明串行处理。PhotonInfer 实现了**真正的并行批处理**，线性扩展。

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
