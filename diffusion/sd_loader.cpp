#include "ggml.h"
#include "gguf.h"
#include <stdexcept>
#include <string>
#include <unordered_map>

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

static std::string kv_to_str(const gguf_context * uf, int i) {
    const gguf_type type = gguf_get_kv_type(uf, i);
    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(uf, i);
        case GGUF_TYPE_UINT8:
            return std::to_string((int)gguf_get_val_u8(uf, i));
        case GGUF_TYPE_INT8:
            return std::to_string((int)gguf_get_val_i8(uf, i));
        case GGUF_TYPE_UINT16:
            return std::to_string((int)gguf_get_val_u16(uf, i));
        case GGUF_TYPE_INT16:
            return std::to_string((int)gguf_get_val_i16(uf, i));
        case GGUF_TYPE_UINT32:
            return std::to_string(gguf_get_val_u32(uf, i));
        case GGUF_TYPE_INT32:
            return std::to_string(gguf_get_val_i32(uf, i));
        case GGUF_TYPE_UINT64:
            return std::to_string((long long)gguf_get_val_u64(uf, i));
        case GGUF_TYPE_INT64:
            return std::to_string((long long)gguf_get_val_i64(uf, i));
        case GGUF_TYPE_FLOAT32:
            return std::to_string(gguf_get_val_f32(uf, i));
        case GGUF_TYPE_FLOAT64:
            return std::to_string(gguf_get_val_f64(uf, i));
        case GGUF_TYPE_BOOL:
            return gguf_get_val_bool(uf, i) ? "true" : "false";
        default:
            return std::string();
    }
}

Model load(const std::string & path) {
    ggml_context * ctx = nullptr;
    gguf_init_params params { true, &ctx };
    gguf_context * uf = gguf_init_from_file(path.c_str(), params);
    if (!uf) throw std::runtime_error("failed to open gguf");

    Model m;
    m.ctx = ctx;
    m.uf = uf;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        m.tensors.emplace(std::string(ggml_get_name(t)), t);
    }

    const int n_kv = gguf_get_n_kv(uf);
    m.kv.reserve((size_t)n_kv);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(uf, i);
        const gguf_type type = gguf_get_kv_type(uf, i);
        if (type == GGUF_TYPE_ARRAY) continue;
        m.kv.emplace(std::string(key), kv_to_str(uf, i));
    }

    return m;
}

}
