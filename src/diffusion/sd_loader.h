#pragma once

#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;

    Model() = default;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    Model(Model && other) noexcept {
        ctx = other.ctx;
        uf = other.uf;
        tensors = std::move(other.tensors);
        kv = std::move(other.kv);
        other.ctx = nullptr;
        other.uf = nullptr;
    }
    Model & operator=(Model && other) noexcept {
        if (this != &other) {
            release();
            ctx = other.ctx;
            uf = other.uf;
            tensors = std::move(other.tensors);
            kv = std::move(other.kv);
            other.ctx = nullptr;
            other.uf = nullptr;
        }
        return *this;
    }
    ~Model() { release(); }

    void release() {
        if (uf) {
            gguf_free(uf);
            uf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        tensors.clear();
        kv.clear();
    }
};

Model load(const std::string & path);

inline ggml_tensor * get_tensor(const Model & m, const std::string & name) {
    auto it = m.tensors.find(name);
    return it == m.tensors.end() ? nullptr : it->second;
}

inline std::string get_kv(const Model & m, const std::string & key) {
    auto it = m.kv.find(key);
    return it == m.kv.end() ? std::string() : it->second;
}

inline std::vector<std::string> kv_keys_with_prefix(const Model & m, const std::string & prefix) {
    std::vector<std::string> out;
    for (const auto & kv : m.kv) {
        if (kv.first.rfind(prefix, 0) == 0) {
            out.push_back(kv.first);
        }
    }
    return out;
}

}


