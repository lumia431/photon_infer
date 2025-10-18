# ============================================================================
# PhotonInfer Core Library
# ============================================================================

# Collect all source files
file(GLOB_RECURSE PHOTON_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp
)

# Collect CUDA source files if CUDA is enabled
if(PHOTON_BUILD_CUDA)
  file(GLOB_RECURSE PHOTON_CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cu
  )
endif()

# Create the main library
if(PHOTON_BUILD_SHARED)
  add_library(photon_core SHARED
    ${PHOTON_CORE_SOURCES}
    ${PHOTON_CUDA_SOURCES}
  )
else()
  add_library(photon_core STATIC
    ${PHOTON_CORE_SOURCES}
    ${PHOTON_CUDA_SOURCES}
  )
endif()

# Link dependencies
target_link_libraries(photon_core
  PUBLIC
    Eigen3::Eigen
    glog::glog
    Threads::Threads
)

if(PHOTON_BUILD_CUDA)
  target_link_libraries(photon_core
    PUBLIC
      CUDA::cudart
      CUDA::cublas
  )
endif()

# Set target properties
set_target_properties(photon_core PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/photon/core"
)

# Include directories
target_include_directories(photon_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compile definitions
if(PHOTON_BUILD_CUDA)
  target_compile_definitions(photon_core PUBLIC PHOTON_USE_CUDA)
endif()
