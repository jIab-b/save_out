cmake_minimum_required(VERSION 3.18)
project(ggml_diffusion)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
