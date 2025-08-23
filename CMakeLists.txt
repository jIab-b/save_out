cmake_minimum_required(VERSION 3.18)
project(ggml_diffusion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to RelWithDebInfo for line info if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Debugging flags for file:line mapping and better backtraces
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -fno-omit-frame-pointer")

find_package(CUDAToolkit)

option(GGML_CUDA "Enable CUDA backend" ON)
option(GGML_STATIC "Build static libraries" OFF)

if(GGML_CUDA AND CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(GGML_USE_CUDA)
endif()

add_subdirectory(ggml)

# Optional llama_core: build only if headers exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/llama-model.h AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/llama-graph.h)
    add_library(llama_core
        src/llama-model.cpp
        src/llama-model-loader.cpp
        src/llama-model-saver.cpp
        src/llama-arch.cpp
        src/llama-hparams.cpp
        src/llama-vocab.cpp
        src/llama-grammar.cpp
        src/llama-impl.cpp
        src/llama-mmap.cpp
        src/llama-quant.cpp
        src/llama-sampling.cpp
        src/unicode.cpp
        src/unicode-data.cpp
    )

    target_include_directories(llama_core PUBLIC
        include
        ggml/include
        src
    )

    target_link_libraries(llama_core PUBLIC ggml)

    if(GGML_CUDA AND CUDAToolkit_FOUND)
        target_link_libraries(llama_core PUBLIC ggml-cuda)
    endif()
endif()

add_library(diffusion_core
    diffusion/sd_loader.cpp
    diffusion/sd_text.cpp
    diffusion/sd_unet.cpp
    diffusion/sd_vae.cpp
    diffusion/sd_scheduler.cpp
    diffusion/sd_tokenizer.cpp
    diffusion/image_io.cpp
    diffusion/sd_profile.cpp
)

target_include_directories(diffusion_core PUBLIC
    include
    ggml/include
    src
    diffusion
)

target_link_libraries(diffusion_core PUBLIC ggml)

if(GGML_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(diffusion_core PUBLIC ggml-cuda)
endif()

if(TARGET llama_core)
    add_library(common_utils
        common/common.cpp
        common/sampling.cpp
        common/speculative.cpp
        common/ngram-cache.cpp
    )

    target_include_directories(common_utils PUBLIC
        include
        ggml/include
        common
    )

    target_link_libraries(common_utils PUBLIC llama_core)
endif()

add_executable(diffuse diffusion/diffuse.cpp)
target_include_directories(diffuse PUBLIC include src diffusion ggml/include)
target_link_libraries(diffuse PUBLIC diffusion_core)
target_link_options(diffuse PRIVATE -rdynamic)

# Text inspector for tests
