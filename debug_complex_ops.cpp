/**
 * @file debug_complex_ops.cpp
 * @brief Debug complex CUDA operators: RoPE, MHA, SwiGLU
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "photon/core/tensor.hpp"
#include "photon/ops/rope.hpp"
#include "photon/ops/mha.hpp"
#include "photon/ops/swiglu.hpp"

using namespace photon;

// Helper: compare tensors
bool tensors_match(const Tensor& cpu_out, const Tensor& cuda_out,
                   f32 rtol = 1e-3f, f32 atol = 1e-4f, bool verbose = false) {
  if (cpu_out.size() != cuda_out.size()) {
    std::cerr << "Size mismatch: " << cpu_out.size() << " vs " << cuda_out.size() << "\n";
    return false;
  }

  auto cuda_cpu = cuda_out.to(DeviceType::CPU);
  if (!cuda_cpu) {
    std::cerr << "Failed to copy CUDA output to CPU\n";
    return false;
  }

  auto cpu_map = cpu_out.vector_map<f32>();
  auto cuda_map = cuda_cpu.value().vector_map<f32>();

  int mismatches = 0;
  f32 max_diff = 0.0f;
  f32 sum_sq_diff = 0.0f;

  for (usize i = 0; i < cpu_out.size(); ++i) {
    f32 diff = std::abs(cpu_map[i] - cuda_map[i]);
    f32 threshold = atol + rtol * std::abs(cpu_map[i]);

    sum_sq_diff += diff * diff;
    if (diff > max_diff) max_diff = diff;

    if (diff > threshold) {
      if (verbose && mismatches < 10) {
        std::cout << "  Mismatch[" << i << "]: CPU=" << cpu_map[i]
                  << " CUDA=" << cuda_map[i] << " diff=" << diff << " threshold=" << threshold << "\n";
      }
      mismatches++;
    }
  }

  f32 rmse = std::sqrt(sum_sq_diff / cpu_out.size());

  if (mismatches > 0) {
    std::cout << "  ✗ " << mismatches << "/" << cpu_out.size()
              << " mismatches (max diff: " << max_diff << ", RMSE: " << rmse << ")\n";
    return false;
  }

  std::cout << "  ✓ All values match (max diff: " << max_diff << ", RMSE: " << rmse << ")\n";
  return true;
}

int main() {
  std::cout << "=== Debugging Complex CUDA Operators ===\n\n";

  // Check CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices!\n";
    return 1;
  }
  std::cout << "✓ CUDA available\n\n";

  const i32 dim = 128;
  const i32 head_num = 4;
  const i32 kv_head_num = 2;
  const i32 head_size = 32;
  const i32 kv_dim = kv_head_num * head_size;
  const i32 seq_len = 64;
  const i32 pos = 10;

  // =================================================================
  // Test 1: RoPE
  // =================================================================
  std::cout << "Test 1: RoPE\n";
  std::cout << "------------\n";

  RoPEOp rope_cpu(dim, kv_dim, head_size, seq_len);
  RoPEOp rope_cuda(dim, kv_dim, head_size, seq_len);
  rope_cuda.set_device(DeviceType::CUDA);

  if (!rope_cpu.init() || !rope_cuda.init()) {
    std::cerr << "  ✗ Failed to init RoPE ops\n";
    return 1;
  }

  // Create Q and K
  auto q_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU).value();
  auto k_cpu = Tensor::zeros({kv_dim}, DataType::Float32, DeviceType::CPU).value();

  auto q_map = q_cpu.vector_map<f32>();
  auto k_map = k_cpu.vector_map<f32>();
  for (i32 i = 0; i < dim; ++i) {
    q_map[i] = 0.1f * static_cast<f32>(i);
  }
  for (i32 i = 0; i < kv_dim; ++i) {
    k_map[i] = 0.05f * static_cast<f32>(i);
  }

  auto q_cuda = q_cpu.to(DeviceType::CUDA).value();
  auto k_cuda = k_cpu.to(DeviceType::CUDA).value();

  // Apply RoPE
  std::cout << "  Running CPU RoPE (pos=" << pos << ")...\n";
  if (!rope_cpu.forward(q_cpu, k_cpu, pos)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA RoPE (pos=" << pos << ")...\n";
  if (!rope_cuda.forward(q_cuda, k_cuda, pos)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare Q
  std::cout << "  Comparing Q outputs...\n";
  if (!tensors_match(q_cpu, q_cuda, 1e-3f, 1e-4f, true)) {
    std::cout << "✗ RoPE Q test FAILED\n\n";
    return 1;
  }

  // Compare K
  std::cout << "  Comparing K outputs...\n";
  if (!tensors_match(k_cpu, k_cuda, 1e-3f, 1e-4f, true)) {
    std::cout << "✗ RoPE K test FAILED\n\n";
    return 1;
  }

  std::cout << "✓ RoPE test PASSED\n\n";

  // =================================================================
  // Test 2: SwiGLU
  // =================================================================
  std::cout << "Test 2: SwiGLU\n";
  std::cout << "--------------\n";

  const i32 hidden_dim = 256;

  SwiGLUOp swiglu_cpu(hidden_dim);
  SwiGLUOp swiglu_cuda(hidden_dim);
  swiglu_cuda.set_device(DeviceType::CUDA);

  if (!swiglu_cpu.init() || !swiglu_cuda.init()) {
    std::cerr << "  ✗ Failed to init SwiGLU ops\n";
    return 1;
  }

  // Create inputs
  auto in1_cpu = Tensor::zeros({hidden_dim}, DataType::Float32, DeviceType::CPU).value();
  auto in2_cpu = Tensor::zeros({hidden_dim}, DataType::Float32, DeviceType::CPU).value();

  auto in1_map = in1_cpu.vector_map<f32>();
  auto in2_map = in2_cpu.vector_map<f32>();
  for (i32 i = 0; i < hidden_dim; ++i) {
    in1_map[i] = 0.02f * static_cast<f32>(i - 128);  // Range: -2.56 to 2.54
    in2_map[i] = 0.01f * static_cast<f32>(i);        // Range: 0 to 2.55
  }

  auto in1_cuda = in1_cpu.to(DeviceType::CUDA).value();
  auto in2_cuda = in2_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto swiglu_out_cpu = Tensor::create({hidden_dim}, DataType::Float32, DeviceType::CPU).value();
  auto swiglu_out_cuda = Tensor::create({hidden_dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "  Running CPU SwiGLU...\n";
  if (!swiglu_cpu.forward(in1_cpu, in2_cpu, swiglu_out_cpu)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA SwiGLU...\n";
  if (!swiglu_cuda.forward(in1_cuda, in2_cuda, swiglu_out_cuda)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare
  std::cout << "  Comparing outputs...\n";
  if (!tensors_match(swiglu_out_cpu, swiglu_out_cuda, 1e-3f, 1e-4f, true)) {
    std::cout << "✗ SwiGLU test FAILED\n\n";
    return 1;
  }
  std::cout << "✓ SwiGLU test PASSED\n\n";

  // =================================================================
  // Test 3: MHA (simplified)
  // =================================================================
  std::cout << "Test 3: MHA\n";
  std::cout << "-----------\n";

  MHAOp mha_cpu(dim, kv_dim, head_num, head_size, seq_len);
  MHAOp mha_cuda(dim, kv_dim, head_num, head_size, seq_len);
  mha_cuda.set_device(DeviceType::CUDA);

  if (!mha_cpu.init() || !mha_cuda.init()) {
    std::cerr << "  ✗ Failed to init MHA ops\n";
    return 1;
  }

  // Create Q, K, V
  auto mha_q_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU).value();
  auto mha_k_cache_cpu = Tensor::zeros({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU).value();
  auto mha_v_cache_cpu = Tensor::zeros({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU).value();

  auto mha_q_map = mha_q_cpu.vector_map<f32>();
  auto mha_k_map = mha_k_cache_cpu.vector_map<f32>();
  auto mha_v_map = mha_v_cache_cpu.vector_map<f32>();

  // Initialize Q
  for (i32 i = 0; i < dim; ++i) {
    mha_q_map[i] = 0.01f * static_cast<f32>(i);
  }

  // Initialize K and V caches (fill first 'pos+1' positions)
  for (i32 t = 0; t <= pos; ++t) {
    for (i32 i = 0; i < kv_dim; ++i) {
      mha_k_map[t * kv_dim + i] = 0.01f * static_cast<f32>(t + i);
      mha_v_map[t * kv_dim + i] = 0.02f * static_cast<f32>(t * 10 + i);
    }
  }

  auto mha_q_cuda = mha_q_cpu.to(DeviceType::CUDA).value();
  auto mha_k_cache_cuda = mha_k_cache_cpu.to(DeviceType::CUDA).value();
  auto mha_v_cache_cuda = mha_v_cache_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto mha_out_cpu = Tensor::create({dim}, DataType::Float32, DeviceType::CPU).value();
  auto mha_out_cuda = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "  Running CPU MHA (pos=" << pos << ")...\n";
  if (!mha_cpu.forward(mha_q_cpu, mha_k_cache_cpu, mha_v_cache_cpu, mha_out_cpu, pos)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA MHA (pos=" << pos << ")...\n";
  if (!mha_cuda.forward(mha_q_cuda, mha_k_cache_cuda, mha_v_cache_cuda, mha_out_cuda, pos)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare
  std::cout << "  Comparing outputs...\n";
  if (!tensors_match(mha_out_cpu, mha_out_cuda, 1e-2f, 1e-3f, true)) {
    std::cout << "✗ MHA test FAILED\n\n";

    // Print some debugging info
    auto cpu_map = mha_out_cpu.vector_map<f32>();
    auto cuda_cpu_result = mha_out_cuda.to(DeviceType::CUDA).value();
    auto cuda_map = cuda_cpu_result.vector_map<f32>();

    std::cout << "First 20 values:\n";
    std::cout << "  CPU:  ";
    for (int i = 0; i < 20; ++i) std::cout << cpu_map[i] << " ";
    std::cout << "\n  CUDA: ";
    for (int i = 0; i < 20; ++i) std::cout << cuda_map[i] << " ";
    std::cout << "\n\n";

    return 1;
  }
  std::cout << "✓ MHA test PASSED\n\n";

  std::cout << "=== All complex operator tests PASSED ===\n";
  return 0;
}
