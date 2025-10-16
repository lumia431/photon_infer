#include "photon/model/raw_model_data.hpp"
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
