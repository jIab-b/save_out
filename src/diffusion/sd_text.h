#pragma once

#include "sd_loader.h"

namespace sd {

struct TextEncoderConfig {
    int32_t hidden_size = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t max_position_embeddings = 0;
};

class TextEncoder {
public:
    static TextEncoder from_file(const std::string & path);

    ggml_tensor * build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const;
    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * token_ids) const;

    const TextEncoderConfig & config() const { return cfg; }
    const Model & model() const { return m; }

private:
    Model m;
    TextEncoderConfig cfg;

    ggml_tensor * find_token_embedding() const;
    ggml_tensor * find_position_embedding() const;
    static int32_t get_int_kv(const Model & m, const std::string & key, int32_t def);
};

}


