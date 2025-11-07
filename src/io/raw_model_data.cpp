/*
 * Copyright (c) 2025 Lummy
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for full details.
 */

#include "photon/io/raw_model_data.hpp"
#include <sys/mman.h>
#include <unistd.h>

namespace photon::model {

RawModelData::~RawModelData() {
  // Unmap memory if valid
  if (data_ != nullptr && data_ != MAP_FAILED) {
    munmap(data_, file_size_);
    data_ = nullptr;
  }

  // Close file descriptor
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

} // namespace photon::model
