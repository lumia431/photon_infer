/**
 * @file add_kernel.hpp
 * @brief CPU kernels for element-wise addition
 * @version 0.1.0
 */

#ifndef PHOTON_OPS_KERNELS_ADD_KERNEL_HPP
#define PHOTON_OPS_KERNELS_ADD_KERNEL_HPP

#include <photon/core/error.hpp>
#include <photon/core/types.hpp>

#include <span>
#include <Eigen/Core>

namespace photon::kernels {

/**
 * @brief Element-wise addition kernel (naive implementation)
 *
 * Computes: output[i] = input1[i] + input2[i] for all i
 *
 * @param input1 First input vector
 * @param input2 Second input vector
 * @param output Output vector
 * @param len Vector length
 */
template <FloatingPoint T>
void add_naive(std::span<const T> input1, std::span<const T> input2,
               std::span<T> output, i32 len) noexcept {
  for (i32 i = 0; i < len; ++i) {
    output[i] = input1[i] + input2[i];
  }
}

/**
 * @brief Element-wise addition kernel (Eigen implementation)
 *
 * Vectorized version using Eigen array operations for SIMD acceleration
 */
template <FloatingPoint T>
Result<void> add_eigen(std::span<const T> input1, std::span<const T> input2,
                       std::span<T> output, i32 len) {
  using ArrayType = Eigen::Array<T, Eigen::Dynamic, 1>;
  using MapType = Eigen::Map<const ArrayType>;
  using MapTypeMut = Eigen::Map<ArrayType>;

  MapType input1_arr(input1.data(), len);
  MapType input2_arr(input2.data(), len);
  MapTypeMut output_arr(output.data(), len);

  // Vectorized addition
  output_arr = input1_arr + input2_arr;

  return Ok();
}

}  // namespace photon::kernels

#endif  // PHOTON_OPS_KERNELS_ADD_KERNEL_HPP
