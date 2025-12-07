/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

/**
 * @file add.cpp
 * @brief Element-wise addition operator implementation
 * @version 0.1.0
 */

#include "photon/ops/add.hpp"
#include "photon/ops/kernels/add_kernel.hpp"
#include <span>

#ifdef PHOTON_USE_CUDA
#include "photon/ops/kernels/cuda/add_kernel.cuh"
#endif

namespace photon {

AddOp::AddOp(bool use_naive)
    : use_naive_(use_naive) {
  device_ = DeviceType::CPU;
}

Result<void> AddOp::forward(const Tensor& input1, const Tensor& input2, Tensor& output) {
  // Validate inputs
  if (input1.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input1 tensor is empty");
  }
  if (input2.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Input2 tensor is empty");
  }
  if (output.empty()) {
    return Err<void>(ErrorCode::InvalidArgument, "Output tensor is empty");
  }

  // Validate shapes match
  if (input1.size() != input2.size()) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Input tensors size mismatch: input1 size=" +
                std::to_string(input1.size()) + ", input2 size=" +
                std::to_string(input2.size()));
  }

  if (input1.size() != output.size()) {
    return Err<void>(ErrorCode::ShapeMismatch,
                "Output tensor size mismatch: expected " +
                std::to_string(input1.size()) + ", got " +
                std::to_string(output.size()));
  }

  // Validate dtypes match
  if (input1.dtype() != input2.dtype()) {
    return Err<void>(ErrorCode::InvalidDtype,
                "Input tensors dtype mismatch");
  }

  if (input1.dtype() != output.dtype()) {
    return Err<void>(ErrorCode::InvalidDtype,
                "Output tensor dtype mismatch");
  }

  // Validate device compatibility
  if (input1.device() != device_ || input2.device() != device_ ||
      output.device() != device_) {
    return Err<void>(ErrorCode::DeviceMismatch,
                "All tensors must be on the same device");
  }

  // Get tensor size
  i32 len = static_cast<i32>(input1.size());

  // Dispatch to appropriate kernel based on device
  if (device_ == DeviceType::CPU) {
    if (input1.dtype() == DataType::Float32) {
#ifdef PHOTON_USE_EIGEN
      auto input1_map = input1.vector_map<f32>();
      auto input2_map = input2.vector_map<f32>();
      auto output_map = output.vector_map<f32>();

      std::span<const f32> input1_span(input1_map.data(), input1_map.size());
      std::span<const f32> input2_span(input2_map.data(), input2_map.size());
      std::span<f32> output_span(output_map.data(), output_map.size());
#else
      std::span<const f32> input1_span(input1.ptr<f32>(), input1.size());
      std::span<const f32> input2_span(input2.ptr<f32>(), input2.size());
      std::span<f32> output_span(output.ptr<f32>(), output.size());
#endif

      if (use_naive_) {
        kernels::add_naive<f32>(input1_span, input2_span, output_span, len);
      } else {
#ifdef PHOTON_USE_EIGEN
        auto result = kernels::add_eigen<f32>(input1_span, input2_span, output_span, len);
        if (!result) {
          return result;
        }
#else
        return Err<void>(ErrorCode::NotImplemented,
                        "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
      }
    } else if (input1.dtype() == DataType::Float64) {
#ifdef PHOTON_USE_EIGEN
      auto input1_map = input1.vector_map<f64>();
      auto input2_map = input2.vector_map<f64>();
      auto output_map = output.vector_map<f64>();

      std::span<const f64> input1_span(input1_map.data(), input1_map.size());
      std::span<const f64> input2_span(input2_map.data(), input2_map.size());
      std::span<f64> output_span(output_map.data(), output_map.size());
#else
      std::span<const f64> input1_span(input1.ptr<f64>(), input1.size());
      std::span<const f64> input2_span(input2.ptr<f64>(), input2.size());
      std::span<f64> output_span(output.ptr<f64>(), output.size());
#endif

      if (use_naive_) {
        kernels::add_naive<f64>(input1_span, input2_span, output_span, len);
      } else {
#ifdef PHOTON_USE_EIGEN
        auto result = kernels::add_eigen<f64>(input1_span, input2_span, output_span, len);
        if (!result) {
          return result;
        }
#else
        return Err<void>(ErrorCode::NotImplemented,
                        "Eigen implementation not available - rebuild with PHOTON_USE_EIGEN=ON");
#endif
      }
    } else {
      return Err<void>(ErrorCode::InvalidArgument,
                  "Unsupported data type for Add operation");
    }
  }
#ifdef PHOTON_USE_CUDA
  else if (device_ == DeviceType::CUDA) {
    // CUDA path (using standard approach)
    if (input1.dtype() == DataType::Float32) {
      std::span<const f32> input1_span(input1.ptr<f32>(), input1.size());
      std::span<const f32> input2_span(input2.ptr<f32>(), input2.size());
      std::span<f32> output_span(output.ptr<f32>(), output.size());

      return kernels::cuda::add_cuda_launch(
          input1_span, input2_span, output_span, len, nullptr);
    } else {
      return Err<void>(ErrorCode::InvalidArgument,
                  "CUDA Add only supports Float32");
    }
  }
#endif
  else {
    return Err<void>(ErrorCode::NotImplemented,
                "Add operation not implemented for this device");
  }

  return Ok();
}

}  // namespace photon
