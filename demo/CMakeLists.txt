# ============================================================================
# PhotonInfer Demo Executables
# ============================================================================

# llama_infer demo executable
add_executable(llama_infer
  llama_infer.cpp
)

# Link against photon_core library
target_link_libraries(llama_infer
  PRIVATE
    photon_core
)

# Include directories
target_include_directories(llama_infer
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(llama_infer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS llama_infer
  RUNTIME DESTINATION bin
)

# batched_infer_multi_prompts demo executable
add_executable(batched_infer_multi_prompts
  batched_infer_multi_prompts.cpp
)

# Link against photon_core library
target_link_libraries(batched_infer_multi_prompts
  PRIVATE
    photon_core
)

# Include directories
target_include_directories(batched_infer_multi_prompts
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(batched_infer_multi_prompts PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS batched_infer_multi_prompts
  RUNTIME DESTINATION bin
)

# compare_batching_methods demo executable
add_executable(compare_batching_methods
  compare_batching_methods.cpp
)

# Link against photon_core library
target_link_libraries(compare_batching_methods
  PRIVATE
    photon_core
)

# Include directories
target_include_directories(compare_batching_methods
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(compare_batching_methods PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS compare_batching_methods
  RUNTIME DESTINATION bin
)
