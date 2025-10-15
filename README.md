# PhotonInfer

A modern C++20 deep learning inference framework for large language models, built from scratch with performance and clarity in mind.

## 🌟 Features

- **Modern C++20**: Leverages Concepts, Ranges, `std::span`, and other C++20 features
- **Type-Safe Error Handling**: Rust-inspired `Result<T, E>` type for explicit error propagation
- **Device Agnostic**: Unified interface for CPU and CUDA backends
- **Zero-Copy Operations**: Extensive use of `std::span` and move semantics
- **Comprehensive Testing**: Full Google Test coverage for all components
- **Clean Architecture**: Minimal dependencies, clear separation of concerns

## 🚀 Getting Started

### Prerequisites

- **Compiler**: GCC 12+ or Clang 14+ (with C++20 support)
- **CMake**: 3.20 or higher
- **CUDA Toolkit**: 11.0+ (optional, for GPU support)
- **Dependencies**:
  - Google Test (for unit tests)
  - Google Log (glog)

### Building

```bash
# Clone the repository
cd photon_infer

# Create build directory
mkdir build && cd build

# Configure (CPU only)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Or with CUDA support
cmake -DCMAKE_BUILD_TYPE=Release -DPHOTON_BUILD_CUDA=ON ..

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `PHOTON_BUILD_TESTS` | Build unit tests | ON |
| `PHOTON_BUILD_EXAMPLES` | Build examples | ON |
| `PHOTON_BUILD_CUDA` | Enable CUDA support | ON |
| `PHOTON_BUILD_SHARED` | Build shared libraries | OFF |
| `PHOTON_ENABLE_ASAN` | Enable AddressSanitizer | OFF |
| `PHOTON_ENABLE_UBSAN` | Enable UndefinedBehaviorSanitizer | OFF |

## 📚 Project Structure

```
photon_infer/
├── include/photon/          # Public headers
│   ├── core/                # Core components
│   │   ├── types.hpp        # Type definitions and concepts
│   │   ├── error.hpp        # Error handling (Result<T>)
│   │   ├── allocator.hpp    # Memory allocators
│   │   └── buffer.hpp       # Memory buffers
│   ├── ops/                 # Operators
│   ├── model/               # Model definitions
│   └── utils/               # Utilities
├── src/                     # Implementation files
│   ├── core/
│   ├── ops/
│   │   ├── cpu/            # CPU implementations
│   │   └── cuda/           # CUDA implementations
│   └── model/
├── tests/                   # Unit tests
├── examples/                # Example programs
├── tools/                   # Utility scripts
└── docs/                    # Documentation
```

## 🎯 Roadmap

This project is being developed incrementally with well-defined PRs:

### Phase 1: Foundation (PR1) ✅
- [x] Project structure and CMake configuration
- [x] Core type system with C++20 Concepts
- [x] Error handling system (`Result<T, E>`)
- [x] Memory allocators (CPU + CUDA)
- [x] Buffer abstraction
- [x] Comprehensive unit tests

### Phase 2: Tensor Operations (PR2-4)
- [ ] Tensor class with shape management
- [ ] Basic tensor operations
- [ ] Model file I/O (mmap-based)
- [ ] TikToken tokenizer implementation

### Phase 3: CPU Operators (PR5-10)
- [ ] Embedding layer
- [ ] Matrix multiplication (GEMV/GEMM)
- [ ] RMSNorm
- [ ] RoPE (Rotary Position Embedding)
- [ ] Multi-Head Attention
- [ ] SwiGLU activation

### Phase 4: Model Inference (PR11-12)
- [ ] LLaMA model structure
- [ ] KV-Cache mechanism
- [ ] Inference pipeline
- [ ] Text generation with sampling

### Phase 5: CUDA Acceleration (PR13-19)
- [ ] CUDA kernels for all operators
- [ ] Memory transfer optimizations
- [ ] Kernel fusion opportunities

### Phase 6: Quantization (PR20-22)
- [ ] Int8 quantization
- [ ] Quantized matrix multiplication
- [ ] Model conversion tools

## 🔧 Code Examples

### Error Handling with Result<T>

```cpp
#include "photon/core/error.hpp"

using namespace photon;

// Function that might fail
Result<int> divide(int a, int b) {
  if (b == 0) {
    return Err<int>(ErrorCode::InvalidArgument, "Division by zero");
  }
  return Ok(a / b);
}

// Usage
auto result = divide(10, 2);
if (result) {
  std::cout << "Result: " << result.value() << std::endl;
} else {
  std::cerr << "Error: " << result.error().to_string() << std::endl;
}
```

### Memory Management with Buffer

```cpp
#include "photon/core/buffer.hpp"

using namespace photon;

// Create a buffer
auto result = Buffer::create(1024, DeviceType::CPU);
if (!result) {
  // Handle error
  return;
}

Buffer buffer = std::move(result.value());

// Access as typed span (zero-copy)
auto span = buffer.as_span<float>();
for (size_t i = 0; i < span.size(); ++i) {
  span[i] = static_cast<float>(i);
}

// Buffer is automatically freed when it goes out of scope
```

### Using C++20 Concepts

```cpp
#include "photon/core/types.hpp"

using namespace photon;

// Generic function constrained by Concept
template <FloatingPoint T>
T compute_mean(std::span<const T> data) {
  T sum = 0;
  for (T value : data) {
    sum += value;
  }
  return sum / static_cast<T>(data.size());
}

// Only accepts floating point types
float result1 = compute_mean<float>({1.0f, 2.0f, 3.0f});
// int result2 = compute_mean<int>({1, 2, 3});  // Compile error!
```

## 🧪 Testing

Run all tests:
```bash
cd build
ctest --output-on-failure
```

Run specific test suite:
```bash
./bin/photon_tests --gtest_filter=BufferTest.*
```

With verbose output:
```bash
./bin/photon_tests --gtest_filter=BufferTest.* --gtest_print_time=1
```

## 📖 Documentation

- [Design Philosophy](docs/design.md)
- [API Reference](docs/api.md)
- [Contributing Guide](docs/contributing.md)
- [Performance Guide](docs/performance.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama2.c](https://github.com/karpathy/llama2.c)
- Built on top of the excellent KuiperInfer educational framework
- Uses design patterns from Rust's error handling

## 📬 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project designed to teach deep learning inference from scratch. It prioritizes clarity and modern C++ practices over absolute performance.
