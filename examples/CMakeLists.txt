# ============================================================================
# PhotonInfer Examples
# ============================================================================

# Example: Basic usage of PR1 features
add_executable(example_basic example_basic.cpp)

target_link_libraries(example_basic
  PRIVATE
    photon_core
)

target_include_directories(example_basic
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

message(STATUS "Examples configured: example_basic")
