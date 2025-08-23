// Shared model types for diffusion components
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;

    Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    Model(Model && other) noexcept;
    Model & operator=(Model && other) noexcept;
    ~Model();

    void release();
};

Model load(const std::string & path);

} // namespace sd

