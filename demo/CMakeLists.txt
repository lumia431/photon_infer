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
