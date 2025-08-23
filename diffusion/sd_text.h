// Text encoder public API
#pragma once

#include "sd_model.h"
#include <stdexcept>

#include <string>

namespace sd {

struct TextEncoderConfig {
    int32_t hidden_size = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t max_position_embeddings = 0;
    float   layer_norm_eps = 1e-5f;
};

class TextEncoder {
public:
    struct Outputs {
        ggml_tensor * last_hidden_state = nullptr;   // [C, T]
        ggml_tensor * pooled_output = nullptr;       // [C, 1]
        ggml_tensor * pooled_projected = nullptr;    // [P, 1] if present
    };

    static TextEncoder from_file(const std::string & path);

    ggml_tensor * build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * token_ids) const;
    Outputs forward_all(ggml_context * ctx, ggml_tensor * token_ids) const;

    const TextEncoderConfig & config() const;
    const Model & model() const;

    ggml_tensor * find_text_projection() const; // optional (present in text2)

private:
    Model m;
    TextEncoderConfig cfg;

    ggml_tensor * find_token_embedding() const;
    ggml_tensor * find_position_embedding() const;
    ggml_tensor * find_final_ln_weight() const;
    ggml_tensor * find_final_ln_bias() const;
    static int32_t get_int_kv(const Model & m, const std::string & key, int32_t def);
    static float   get_float_kv(const Model & m, const std::string & key, float def);
    static std::string get_str_kv(const Model & m, const std::string & key, const std::string & def);
    ggml_tensor * get(const std::string & name) const {
        auto it = m.tensors.find(name);
        if (it == m.tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
};

// helper to detect ODR/ABI issues at runtime
size_t textencoder_sizeof();

} // namespace sd
