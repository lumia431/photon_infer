/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file test_partitioned_attention.cpp
 * @brief Test partitioned attention performance
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

void generate_random_data(Tensor& tensor) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);

  f32* data = tensor.ptr<f32>();
  for (usize i = 0; i < tensor.size(); ++i) {
    data[i] = dist(gen);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LOG(INFO) << "Partitioned Attention Performance Test\n";

  const i32 batch_size = 16;
  const i32 num_heads = 16;
  const i32 head_size = 64;
  const i32 seq_len = 2048;
  const i32 kv_dim = 512;
  const i32 kv_mul = 2;
  const i32 layer_index = 0;

  std::vector<i32> positions_cpu(batch_size);
  for (i32 i = 0; i < batch_size; ++i) {
    positions_cpu[i] = 50 + i * 100;
  }

  LOG(INFO) << "Batch: " << batch_size << ", Heads: " << num_heads
            << ", SeqLen: " << seq_len << "\n";

  auto query_cpu = Tensor::create({batch_size, num_heads, head_size}, DataType::Float32, DeviceType::CPU);
  auto key_cache_cpu = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  auto value_cache_cpu = Tensor::create({seq_len, kv_dim}, DataType::Float32, DeviceType::CPU);
  auto output_cpu = Tensor::create({batch_size, num_heads, head_size}, DataType::Float32, DeviceType::CPU);

  if (!query_cpu || !key_cache_cpu || !value_cache_cpu || !output_cpu) {
    LOG(ERROR) << "Failed to allocate tensors";
    return 1;
  }

  generate_random_data(query_cpu.value());
  generate_random_data(key_cache_cpu.value());
  generate_random_data(value_cache_cpu.value());

  LOG(INFO) << "Transferring to GPU...";
  auto query_gpu = query_cpu.value().to(DeviceType::CUDA);
  auto key_cache_gpu = key_cache_cpu.value().to(DeviceType::CUDA);
  auto value_cache_gpu = value_cache_cpu.value().to(DeviceType::CUDA);
  auto output_gpu = output_cpu.value().to(DeviceType::CUDA);

  if (!query_gpu || !key_cache_gpu || !value_cache_gpu || !output_gpu) {
    LOG(ERROR) << "Failed to transfer to GPU";
    return 1;
  }

  i32* positions_gpu;
  cudaMalloc(&positions_gpu, batch_size * sizeof(i32));
  cudaMemcpy(positions_gpu, positions_cpu.data(), batch_size * sizeof(i32), cudaMemcpyHostToDevice);

  LOG(INFO) << "Warming up...";
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

  LOG(INFO) << "Benchmarking...";
  const int num_iterations = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    auto result = partitioned_mha_cuda_launch(
        positions_gpu, batch_size, num_heads, layer_index, seq_len, kv_dim, kv_mul, head_size,
        std::span<f32>(output_gpu.value().ptr<f32>(), output_gpu.value().size()),
        std::span<const f32>(query_gpu.value().ptr<f32>(), query_gpu.value().size()),
        std::span<const f32>(key_cache_gpu.value().ptr<f32>(), key_cache_gpu.value().size()),
        std::span<const f32>(value_cache_gpu.value().ptr<f32>(), value_cache_gpu.value().size()),
        nullptr);

    if (!result) {
      LOG(ERROR) << "Failed: " << result.error().message();
      return 1;
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end - start).count();
  double time_per_call = elapsed / num_iterations * 1000.0;

  LOG(INFO) << "\nResults:";
  LOG(INFO) << "  Time per call: " << time_per_call << " ms";
  LOG(INFO) << "  Throughput: " << (num_iterations / elapsed) << " calls/s\n";
  cudaFree(positions_gpu);
  LOG(INFO) << "Test complete!";
  return 0;
}
