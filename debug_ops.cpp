/**
 * @file debug_ops.cpp
 * @brief Debug individual CUDA operators by comparing with CPU
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "photon/core/tensor.hpp"
#include "photon/ops/rmsnorm.hpp"
#include "photon/ops/matmul.hpp"
#include "photon/ops/rope.hpp"
#include "photon/ops/add.hpp"

using namespace photon;

// Helper: compare tensors
bool tensors_match(const Tensor& cpu_out, const Tensor& cuda_out,
                   f32 rtol = 1e-4f, f32 atol = 1e-5f, bool verbose = false) {
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

  for (usize i = 0; i < cpu_out.size(); ++i) {
    f32 diff = std::abs(cpu_map[i] - cuda_map[i]);
    f32 threshold = atol + rtol * std::abs(cpu_map[i]);

    if (diff > max_diff) max_diff = diff;

    if (diff > threshold) {
      if (verbose && mismatches < 10) {
        std::cout << "  Mismatch[" << i << "]: CPU=" << cpu_map[i]
                  << " CUDA=" << cuda_map[i] << " diff=" << diff << "\n";
      }
      mismatches++;
    }
  }

  if (mismatches > 0) {
    std::cout << "  ✗ " << mismatches << " mismatches (max diff: " << max_diff << ")\n";
    return false;
  }

  std::cout << "  ✓ All values match (max diff: " << max_diff << ")\n";
  return true;
}

int main() {
  std::cout << "=== Debugging CUDA Operators ===\n\n";

  // Check CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices!\n";
    return 1;
  }
  std::cout << "✓ CUDA available\n\n";

  const i32 dim = 64;
  const i32 hidden_dim = 256;
  const f32 eps = 1e-5f;

  // =================================================================
  // Test 1: RMSNorm
  // =================================================================
  std::cout << "Test 1: RMSNorm\n";
  std::cout << "---------------\n";

  RMSNormOp rmsnorm_cpu(dim, eps);
  RMSNormOp rmsnorm_cuda(dim, eps);
  rmsnorm_cuda.set_device(DeviceType::CUDA);

  // Create weight
  auto rms_weight_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU);
  auto rms_weight_cuda = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU);
  if (!rms_weight_cpu || !rms_weight_cuda) {
    std::cerr << "  ✗ Failed to create weights\n";
    return 1;
  }

  auto rms_w_cpu_map = rms_weight_cpu.value().vector_map<f32>();
  auto rms_w_cuda_map = rms_weight_cuda.value().vector_map<f32>();
  for (i32 i = 0; i < dim; ++i) {
    f32 val = 1.0f + 0.01f * static_cast<f32>(i);
    rms_w_cpu_map[i] = val;
    rms_w_cuda_map[i] = val;
  }

  if (!rmsnorm_cpu.set_weight(std::move(rms_weight_cpu.value())) || !rmsnorm_cpu.init()) {
    std::cerr << "  ✗ Failed to init CPU RMSNorm\n";
    return 1;
  }
  if (!rmsnorm_cuda.set_weight(std::move(rms_weight_cuda.value())) || !rmsnorm_cuda.init()) {
    std::cerr << "  ✗ Failed to init CUDA RMSNorm\n";
    return 1;
  }

  // Create input
  auto rms_input_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU).value();
  auto rms_input_map = rms_input_cpu.vector_map<f32>();
  for (i32 i = 0; i < dim; ++i) {
    rms_input_map[i] = 0.5f * static_cast<f32>(i - 32);  // Range: -16 to 15.5
  }

  auto rms_input_cuda = rms_input_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto rms_out_cpu = Tensor::create({dim}, DataType::Float32, DeviceType::CPU).value();
  auto rms_out_cuda = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "  Running CPU RMSNorm...\n";
  if (!rmsnorm_cpu.forward(rms_input_cpu, rms_out_cpu)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA RMSNorm...\n";
  if (!rmsnorm_cuda.forward(rms_input_cuda, rms_out_cuda)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare
  std::cout << "  Comparing outputs...\n";
  if (!tensors_match(rms_out_cpu, rms_out_cuda, 1e-4f, 1e-5f, true)) {
    std::cout << "✗ RMSNorm test FAILED\n\n";

    // Print some values for debugging
    auto cpu_map = rms_out_cpu.vector_map<f32>();
    auto cuda_cpu_result = rms_out_cuda.to(DeviceType::CPU).value();
    auto cuda_map = cuda_cpu_result.vector_map<f32>();

    std::cout << "First 10 values:\n";
    std::cout << "  CPU:  ";
    for (int i = 0; i < 10; ++i) std::cout << cpu_map[i] << " ";
    std::cout << "\n  CUDA: ";
    for (int i = 0; i < 10; ++i) std::cout << cuda_map[i] << " ";
    std::cout << "\n\n";

    return 1;
  }
  std::cout << "✓ RMSNorm test PASSED\n\n";

  // =================================================================
  // Test 2: MatMul
  // =================================================================
  std::cout << "Test 2: MatMul\n";
  std::cout << "--------------\n";

  const i32 in_dim = 64;
  const i32 out_dim = 128;

  MatMulOp matmul_cpu(in_dim, out_dim);
  MatMulOp matmul_cuda(in_dim, out_dim);
  matmul_cuda.set_device(DeviceType::CUDA);

  // Create weight [out_dim × in_dim]
  auto mm_weight_cpu = Tensor::zeros({out_dim, in_dim}, DataType::Float32, DeviceType::CPU);
  auto mm_weight_cuda = Tensor::zeros({out_dim, in_dim}, DataType::Float32, DeviceType::CPU);
  if (!mm_weight_cpu || !mm_weight_cuda) {
    std::cerr << "  ✗ Failed to create weights\n";
    return 1;
  }

  auto mm_w_cpu_map = mm_weight_cpu.value().matrix_map<f32>();
  auto mm_w_cuda_map = mm_weight_cuda.value().matrix_map<f32>();
  for (i32 i = 0; i < out_dim; ++i) {
    for (i32 j = 0; j < in_dim; ++j) {
      f32 val = 0.01f * static_cast<f32>(i - j);
      mm_w_cpu_map(i, j) = val;
      mm_w_cuda_map(i, j) = val;
    }
  }

  if (!matmul_cpu.set_weight(std::move(mm_weight_cpu.value())) || !matmul_cpu.init()) {
    std::cerr << "  ✗ Failed to init CPU MatMul\n";
    return 1;
  }
  if (!matmul_cuda.set_weight(std::move(mm_weight_cuda.value())) || !matmul_cuda.init()) {
    std::cerr << "  ✗ Failed to init CUDA MatMul\n";
    return 1;
  }

  // Create input
  auto mm_input_cpu = Tensor::zeros({in_dim}, DataType::Float32, DeviceType::CPU).value();
  auto mm_input_map = mm_input_cpu.vector_map<f32>();
  for (i32 i = 0; i < in_dim; ++i) {
    mm_input_map[i] = 0.1f * static_cast<f32>(i);
  }

  auto mm_input_cuda = mm_input_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto mm_out_cpu = Tensor::create({out_dim}, DataType::Float32, DeviceType::CPU).value();
  auto mm_out_cuda = Tensor::create({out_dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "  Running CPU MatMul...\n";
  if (!matmul_cpu.forward(mm_input_cpu, mm_out_cpu)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA MatMul...\n";
  if (!matmul_cuda.forward(mm_input_cuda, mm_out_cuda)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare
  std::cout << "  Comparing outputs...\n";
  if (!tensors_match(mm_out_cpu, mm_out_cuda, 1e-3f, 1e-4f, true)) {
    std::cout << "✗ MatMul test FAILED\n\n";
    return 1;
  }
  std::cout << "✓ MatMul test PASSED\n\n";

  // =================================================================
  // Test 3: Add
  // =================================================================
  std::cout << "Test 3: Add\n";
  std::cout << "-----------\n";

  AddOp add_cpu;
  AddOp add_cuda;
  add_cuda.set_device(DeviceType::CUDA);

  if (!add_cpu.init() || !add_cuda.init()) {
    std::cerr << "  ✗ Failed to init Add ops\n";
    return 1;
  }

  // Create inputs
  auto add_in1_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU).value();
  auto add_in2_cpu = Tensor::zeros({dim}, DataType::Float32, DeviceType::CPU).value();
  auto add_in1_map = add_in1_cpu.vector_map<f32>();
  auto add_in2_map = add_in2_cpu.vector_map<f32>();
  for (i32 i = 0; i < dim; ++i) {
    add_in1_map[i] = static_cast<f32>(i);
    add_in2_map[i] = static_cast<f32>(100 - i);
  }

  auto add_in1_cuda = add_in1_cpu.to(DeviceType::CUDA).value();
  auto add_in2_cuda = add_in2_cpu.to(DeviceType::CUDA).value();

  // Create outputs
  auto add_out_cpu = Tensor::create({dim}, DataType::Float32, DeviceType::CPU).value();
  auto add_out_cuda = Tensor::create({dim}, DataType::Float32, DeviceType::CUDA).value();

  // Forward
  std::cout << "  Running CPU Add...\n";
  if (!add_cpu.forward(add_in1_cpu, add_in2_cpu, add_out_cpu)) {
    std::cerr << "  ✗ CPU forward failed\n";
    return 1;
  }

  std::cout << "  Running CUDA Add...\n";
  if (!add_cuda.forward(add_in1_cuda, add_in2_cuda, add_out_cuda)) {
    std::cerr << "  ✗ CUDA forward failed\n";
    return 1;
  }

  // Compare
  std::cout << "  Comparing outputs...\n";
  if (!tensors_match(add_out_cpu, add_out_cuda, 1e-5f, 1e-6f, true)) {
    std::cout << "✗ Add test FAILED\n\n";
    return 1;
  }
  std::cout << "✓ Add test PASSED\n\n";

  std::cout << "=== All operator tests PASSED ===\n";
  return 0;
}
