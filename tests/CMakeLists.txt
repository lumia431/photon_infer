# ============================================================================
# PhotonInfer Unit Tests
# ============================================================================

# Test executable
add_executable(photon_tests
  core/test_types.cpp
  core/test_error.cpp
  core/test_allocator.cpp
  core/test_buffer.cpp
  core/test_tensor.cpp
  core/test_quant.cpp
  model/test_model_loader.cpp
  model/test_tokenizer.cpp
  model/test_llama32_integration.cpp
  model/test_llama_model.cpp
  ops/test_embedding.cpp
  ops/test_matmul.cpp
  ops/test_matmul_quant.cpp
  ops/test_rmsnorm.cpp
  ops/test_rope.cpp
  ops/test_swiglu.cpp
  ops/test_mha.cpp
  ops/test_add.cpp
  test_main.cpp
)

# Link against photon_core and GTest
target_link_libraries(photon_tests
  PRIVATE
    photon_core
    GTest::gtest
    GTest::gtest_main
)

# Include directories
target_include_directories(photon_tests
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Discover tests
gtest_discover_tests(photon_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Add test target
add_test(NAME photon_tests COMMAND photon_tests)

# Add CUDA tests subdirectory
if(PHOTON_BUILD_CUDA)
  add_subdirectory(cuda)

  # ThrustAllocator unit tests (separate executable for focused testing)
  add_executable(test_thrust_allocator
    core/test_thrust_allocator.cpp
  )

  target_link_libraries(test_thrust_allocator
    PRIVATE
      photon_core
      GTest::gtest
      GTest::gtest_main
  )

  target_include_directories(test_thrust_allocator
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
  )

  set_target_properties(test_thrust_allocator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  gtest_discover_tests(test_thrust_allocator
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
