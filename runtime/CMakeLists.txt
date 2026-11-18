# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(yeoubi_runtime C CXX)

# Backends are ggml's own options, passed straight through, e.g.
#   -DGGML_CUDA=ON  -DGGML_HIP=ON  -DGGML_VULKAN=ON  -DGGML_METAL=ON
#   -DGGML_SYCL=ON  -DGGML_OPENVINO=ON  -DGGML_HEXAGON=ON  -DGGML_OPENCL=ON
# Several may be enabled at once; --list-devices shows what was built.
set(YEOUBI_GGML_TAG "v1.0.0" CACHE STRING "ggml git tag to build against")

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt)
    add_subdirectory(ggml)  # local checkout wins (offline / patched builds)
else()
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        ${YEOUBI_GGML_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(ggml)
endif()

add_executable(yeoubi yeoubi.c)
set_target_properties(yeoubi PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(yeoubi PRIVATE ggml)
if (NOT MSVC)
    target_link_libraries(yeoubi PRIVATE m)
endif()
