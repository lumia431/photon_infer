# CUDA operator tests
if(PHOTON_BUILD_CUDA)
  add_executable(cuda_ops_test
    test_cuda_ops.cpp
  )

  target_link_libraries(cuda_ops_test
    PRIVATE
      photon_core
      GTest::gtest
      GTest::gtest_main
      CUDA::cudart
  )

  target_include_directories(cuda_ops_test
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
  )
endif()
