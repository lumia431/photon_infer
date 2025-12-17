# Web Server for PhotonInfer

# Check if dependencies are available locally first
set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")

# Set httplib and nlohmann/json include directories
set(HTTPLIB_INCLUDE_DIR "${DEPS_DIR}")
set(JSON_INCLUDE_DIR "${DEPS_DIR}")

# Web server executable
add_executable(photon_web_server web_server.cpp)

# Find OpenSSL
find_package(OpenSSL REQUIRED)
if(OPENSSL_FOUND)
    add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(photon_web_server
        PRIVATE
            photon_core
            OpenSSL::SSL
            OpenSSL::Crypto
    )
else()
    target_link_libraries(photon_web_server
        PRIVATE
            photon_core
    )
endif()

target_include_directories(photon_web_server
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${HTTPLIB_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
)

# Set output directory
set_target_properties(photon_web_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy static files to build directory
add_custom_command(TARGET photon_web_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/static
        ${CMAKE_BINARY_DIR}/bin/web/static
    COMMENT "Copying web static files to build directory"
)

# Install targets
install(TARGETS photon_web_server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install static files to fixed absolute path
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/static
    DESTINATION /photon_infer/web
)

message(STATUS "Web server will be built as: ${CMAKE_BINARY_DIR}/bin/photon_web_server")
