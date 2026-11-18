# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.19)
project(yeoubi_runtime C CXX)

# Backends are ggml's own options, passed straight through, e.g.
#   -DGGML_CUDA=ON  -DGGML_HIP=ON  -DGGML_VULKAN=ON  -DGGML_METAL=ON
#   -DGGML_SYCL=ON  -DGGML_OPENVINO=ON  -DGGML_HEXAGON=ON  -DGGML_OPENCL=ON
# Several may be enabled at once; --list-devices shows what was built.
set(YEOUBI_GGML_TAG "v1.0.0" CACHE STRING "ggml git tag to build against")
# One self-contained executable: nothing to copy next to it, nothing to install.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Adds a Vulkan kernel for GGML_OP_TOP_K_MASK (cmake/patch_ggml.cmake).
set(PATCH_GGML ${CMAKE_COMMAND} -DGGML_SRC=<SOURCE_DIR> -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_ggml.cmake)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt)
    # Local checkout wins (offline / patched builds). Patching is idempotent.
    execute_process(COMMAND ${CMAKE_COMMAND} -DGGML_SRC=${CMAKE_CURRENT_SOURCE_DIR}/ggml
                            -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_ggml.cmake
                    COMMAND_ERROR_IS_FATAL ANY)
    add_subdirectory(ggml)
else()
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        ${YEOUBI_GGML_TAG}
        GIT_SHALLOW    TRUE
        PATCH_COMMAND  ${PATCH_GGML}
        UPDATE_DISCONNECTED TRUE)  # never re-pull the pinned tag over the patch
    FetchContent_MakeAvailable(ggml)
endif()

add_executable(yeoubi yeoubi.c)
set_target_properties(yeoubi PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(yeoubi PRIVATE ggml)
if (NOT MSVC)
    target_link_libraries(yeoubi PRIVATE m)
endif()
