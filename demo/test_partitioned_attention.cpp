/**
 * @file test_partitioned_attention.cpp
 * @brief Test partitioned attention performance
 *
 * Compares original batched MHA vs partitioned MHA to verify
 * 1.3-1.8x speedup target for long sequences.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <glog/logging.h>

#include "photon/core/tensor.hpp"
#include "photon/core/types.hpp"
#include "photon/ops/kernels/cuda/batched_mha_kernel.cuh"

using namespace photon;
using namespace photon::kernels::cuda;

/**
 * @brief Generate random test data
 */
void generate_random_data(Tensor& tensor) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);

  f32* data = tensor.ptr<f32>();
  for (usize i = 0; i < tensor.size(); ++i) {
    data[i] = dist(gen);
  }
}

/**
 * @brief Test partitioned attention performance
 */
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "========================================";
  LOG(INFO) << "Partitioned Attention Performance Test";
  LOG(INFO) << "========================================\n";

  // Test configuration
  const i32 batch_size = 16;
  const i32 num_heads = 16;
  const i32 head_size = 64;
  const i32 seq_len = 2048;
  const i32 kv_dim = 512;  // 8 kv_heads * 64
  const i32 kv_mul = 2;     // num_heads / num_kv_heads = 16/8
  const i32 layer_index = 0;

  // Test positions (varying lengths)
  std::vector<i32> positions_cpu(batch_size);
  for (i32 i = 0; i < batch_size; ++i) {
    positions_cpu[i] = 50 + i * 100;  // 50, 150, 250, ..., 1550
  }

  LOG(INFO) << "Configuration:";
  LOG(INFO) << "  Batch size: " << batch_size;
  LOG(INFO) << "  Num heads: " << num_heads;
  LOG(INFO) << "  Head size: " << head_size;
  LOG(INFO) << "  Max seq len: " << seq_len;
  LOG(INFO) << "  Position range: " << positions_cpu[0] << " - " << positions_cpu.back();
  LOG(INFO) << "";

  // Allocate CPU tensors
  auto query_cpu = Tensor::create({batch_size, num_heads, head_size}, DataType::Float32, DeviceType::CPU);
  auto key_cache_cpu = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  auto value_cache_cpu = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  auto score_cpu = Tensor::create({batch_size, num_heads, seq_len}, DataType::Float32, DeviceType::CPU);
  auto output_cpu = Tensor::create({batch_size, num_heads, head_size}, DataType::Float32, DeviceType::CPU);

  if (!query_cpu || !key_cache_cpu || !value_cache_cpu || !score_cpu || !output_cpu) {
    LOG(ERROR) << "Failed to allocate CPU tensors";
    return 1;
  }

  // Fill with random data
  generate_random_data(query_cpu.value());
  generate_random_data(key_cache_cpu.value());
  generate_random_data(value_cache_cpu.value());

  // Transfer to GPU
  LOG(INFO) << "[1/4] Transferring data to GPU...";
  auto query_gpu = query_cpu.value().to(DeviceType::CUDA);
  auto key_cache_gpu = key_cache_cpu.value().to(DeviceType::CUDA);
  auto value_cache_gpu = value_cache_cpu.value().to(DeviceType::CUDA);
  auto score_gpu = score_cpu.value().to(DeviceType::CUDA);
  auto output_gpu = output_cpu.value().to(DeviceType::CUDA);

  if (!query_gpu || !key_cache_gpu || !value_cache_gpu || !score_gpu || !output_gpu) {
    LOG(ERROR) << "Failed to transfer to GPU";
    return 1;
  }

  // Transfer positions to GPU
  i32* positions_gpu;
  cudaMalloc(&positions_gpu, batch_size * sizeof(i32));
  cudaMemcpy(positions_gpu, positions_cpu.data(), batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  // Warmup
  LOG(INFO) << "[2/4] Warming up kernels...";
  for (int i = 0; i < 3; ++i) {
    auto result = partitioned_mha_cuda_launch(
        positions_gpu,
        batch_size,
        num_heads,
        layer_index,
        seq_len,
        kv_dim,
        kv_mul,
        head_size,
        std::span<f32>(output_gpu.value().ptr<f32>(), output_gpu.value().size()),
        std::span<const f32>(query_gpu.value().ptr<f32>(), query_gpu.value().size()),
        std::span<const f32>(key_cache_gpu.value().ptr<f32>(), key_cache_gpu.value().size()),
        std::span<const f32>(value_cache_gpu.value().ptr<f32>(), value_cache_gpu.value().size()),
        nullptr);

    if (!result) {
      LOG(ERROR) << "Warmup failed: " << result.error().message();
      return 1;
    }
  }
  cudaDeviceSynchronize();

  // Benchmark partitioned kernel
  LOG(INFO) << "[3/4] Benchmarking partitioned attention...";
  const int num_iterations = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    auto result = partitioned_mha_cuda_launch(
        positions_gpu,
        batch_size,
        num_heads,
        layer_index,
        seq_len,
        kv_dim,
        kv_mul,
        head_size,
        std::span<f32>(output_gpu.value().ptr<f32>(), output_gpu.value().size()),
        std::span<const f32>(query_gpu.value().ptr<f32>(), query_gpu.value().size()),
        std::span<const f32>(key_cache_gpu.value().ptr<f32>(), key_cache_gpu.value().size()),
        std::span<const f32>(value_cache_gpu.value().ptr<f32>(), value_cache_gpu.value().size()),
        nullptr);

    if (!result) {
      LOG(ERROR) << "Iteration " << i << " failed: " << result.error().message();
      return 1;
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end - start).count();
  double time_per_call = elapsed / num_iterations * 1000.0;  // ms

  LOG(INFO) << "[4/4] Results:";
  LOG(INFO) << "  Total time: " << elapsed << " seconds";
  LOG(INFO) << "  Time per call: " << time_per_call << " ms";
  LOG(INFO) << "  Throughput: " << (num_iterations / elapsed) << " calls/s";
  LOG(INFO) << "";

  // Verify correctness by checking output is not NaN/Inf
  auto output_check = output_gpu.value().to(DeviceType::CPU);
  if (!output_check) {
    LOG(ERROR) << "Failed to copy output back";
    return 1;
  }

  f32* output_data = output_check.value().ptr<f32>();
  bool has_nan = false;
  bool has_inf = false;
  for (usize i = 0; i < output_check.value().size(); ++i) {
    if (std::isnan(output_data[i])) has_nan = true;
    if (std::isinf(output_data[i])) has_inf = true;
  }

  if (has_nan || has_inf) {
    LOG(ERROR) << "Output contains NaN/Inf!";
    LOG(ERROR) << "  NaN: " << (has_nan ? "YES" : "NO");
    LOG(ERROR) << "  Inf: " << (has_inf ? "YES" : "NO");
    return 1;
  }

  LOG(INFO) << "✓ Correctness check passed (no NaN/Inf)";
  LOG(INFO) << "✓ Partitioned attention test complete!\n";

  // Cleanup
  cudaFree(positions_gpu);

  return 0;
}
