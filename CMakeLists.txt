cmake_minimum_required(VERSION 3.20)

project(
  PhotonInfer
  VERSION 0.1.0
  DESCRIPTION "Modern C++20 Deep Learning Inference Framework"
  LANGUAGES CXX CUDA
)

# ============================================================================
# Build Options
# ============================================================================
option(PHOTON_BUILD_TESTS "Build unit tests" ON)
option(PHOTON_BUILD_EXAMPLES "Build examples" ON)
option(PHOTON_BUILD_CUDA "Build with CUDA support" ON)
option(PHOTON_BUILD_SHARED "Build shared libraries" OFF)
option(PHOTON_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(PHOTON_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# ============================================================================
# C++ Standard and Compiler Requirements
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for clangd/IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Output Directories
# ============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# Compiler Flags
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wformat=2
  )

  # Release optimizations
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
  endif()

  # Debug symbols
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
  endif()
endif()

# AddressSanitizer
if(PHOTON_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# UndefinedBehaviorSanitizer
if(PHOTON_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined)
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================
if(PHOTON_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CUDA architectures (adjust based on your GPU)
  # Common architectures:
  # - 70: V100
  # - 75: T4
  # - 80: A100
  # - 86: RTX 30xx
  # - 89: RTX 40xx
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  endif()

  message(STATUS "CUDA Enabled: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ============================================================================
# Find Dependencies
# ============================================================================

# Google Test (for unit tests)
if(PHOTON_BUILD_TESTS)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
endif()

# Google Logging (glog)
find_package(glog REQUIRED)

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(src)

if(PHOTON_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(PHOTON_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(
  DIRECTORY include/photon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  PhotonInfer Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Tests: ${PHOTON_BUILD_TESTS}")
message(STATUS "  Build Examples: ${PHOTON_BUILD_EXAMPLES}")
message(STATUS "  Build CUDA: ${PHOTON_BUILD_CUDA}")
message(STATUS "  Build Shared: ${PHOTON_BUILD_SHARED}")
message(STATUS "  Enable ASAN: ${PHOTON_ENABLE_ASAN}")
message(STATUS "  Enable UBSAN: ${PHOTON_ENABLE_UBSAN}")
message(STATUS "========================================")
message(STATUS "")
