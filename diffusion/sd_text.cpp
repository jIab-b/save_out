#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <stdexcept>
#include <cstring>
#include <string>
#include <unordered_map>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

Model load(const std::string & path);

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

    const TextEncoderConfig & config() const;
    const Model & model() const;

private:
    Model m;
    TextEncoderConfig cfg;

    ggml_tensor * find_token_embedding() const;
    ggml_tensor * find_position_embedding() const;
    static int32_t get_int_kv(const Model & m, const std::string & key, int32_t def);
};

const TextEncoderConfig & TextEncoder::config() const { return cfg; }
const Model & TextEncoder::model() const { return m; }

int32_t TextEncoder::get_int_kv(const Model & m, const std::string & key, int32_t def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return def;
    }
}

TextEncoder TextEncoder::from_file(const std::string & path) {
    TextEncoder te;
    te.m = load(path);
    te.cfg.hidden_size = get_int_kv(te.m, "diffusion.text.hidden_size", 0);
    te.cfg.num_layers = get_int_kv(te.m, "diffusion.text.num_hidden_layers", 0);
    te.cfg.num_heads = get_int_kv(te.m, "diffusion.text.num_attention_heads", 0);
    te.cfg.max_position_embeddings = get_int_kv(te.m, "diffusion.text.max_position_embeddings", 0);
    return te;
}

ggml_tensor * TextEncoder::find_token_embedding() const {
    auto it = m.tensors.find("text_model.embeddings.token_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    it = m.tensors.find("model.embeddings.token_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    return nullptr;
}

ggml_tensor * TextEncoder::find_position_embedding() const {
    auto it = m.tensors.find("text_model.embeddings.position_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    it = m.tensors.find("model.embeddings.position_embedding.weight");
    if (it != m.tensors.end()) return it->second;
    return nullptr;
}

ggml_tensor * TextEncoder::build_embeddings(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * tok_src = find_token_embedding();
    if (!tok_src) throw std::runtime_error("token embedding not found in text gguf");
    ggml_tensor * tok = ggml_dup_tensor(ctx, tok_src);
    std::memcpy(tok->data, tok_src->data, ggml_nbytes(tok_src));
    ggml_tensor * ids_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_ids->ne[0]);
    std::memcpy(ids_in->data, token_ids->data, ggml_nbytes(token_ids));
    ggml_tensor * gathered = ggml_get_rows(ctx, tok, ids_in);
    return gathered;
}

ggml_tensor * TextEncoder::forward(ggml_context * ctx, ggml_tensor * token_ids) const {
    ggml_tensor * tok_emb = build_embeddings(ctx, token_ids);
    ggml_tensor * pos_w_src = find_position_embedding();
    if (pos_w_src) {
        ggml_tensor * pos_w = ggml_dup_tensor(ctx, pos_w_src);
        std::memcpy(pos_w->data, pos_w_src->data, ggml_nbytes(pos_w_src));
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_ids->ne[0]);
        int32_t * p = (int32_t *) positions->data;
        for (int i = 0; i < token_ids->ne[0]; ++i) p[i] = i;
        ggml_tensor * pos_emb = ggml_get_rows(ctx, pos_w, positions);
        tok_emb = ggml_add(ctx, tok_emb, pos_emb);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tok_emb);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    return tok_emb;
}

}
