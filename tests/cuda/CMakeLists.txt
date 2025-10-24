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

  # Step-by-step validation test
  add_executable(step_by_step_test
    test_step_by_step.cpp
  )

  target_link_libraries(step_by_step_test
    PRIVATE
      photon_core
      GTest::gtest
      GTest::gtest_main
      CUDA::cudart
  )

  target_include_directories(step_by_step_test
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
  )
endif()
